import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Tuple
import onnx
from onnx import helper
import onnx.checker

def get_attribute(onnx_model, attr_name):
    i = 0
    while 1:
        attributes = onnx_model.graph.node[i].attribute
        for attr in attributes:
            if attr.name == attr_name:
                return attr
        i += 1

class Node:
    def __init__(
            self,
            id,  # 节点id
            feature_id,  # 特征id
            mode,  # 节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
            value,  # 阈值，叶子节点的值为0
            target_id,  # 叶子节点的taget id
            target_weight,  # 叶子节点的权重，即预测值
            samples  # 节点的样本数
            ):
        self.id: int = id
        self.feature_id: int = feature_id
        self.mode: bytes = mode
        self.value: float = value
        self.target_id: int | None = target_id
        self.target_weight: float | None = target_weight
        self.samples: int = samples
        
        self.parent: 'Node' | None = None
        self.left: 'Node' | None = None
        self.right: 'Node' | None  = None

    def branch_samples(self) -> int:
        samples = self.samples
        
        if self.left is not None:
            samples += self.left.branch_samples()
        if self.right is not None:
            samples += self.right.branch_samples()
        
        return samples
    
    def cost(self, alpha: float) -> float:
        if self.mode == b'LEAF':
            return self.samples

        return self.left.cost(alpha) + self.right.cost(alpha) + alpha * self.left.samples  + self.right.samples

    def same_feature_branch_samples(self) -> int:
        samples = self.samples
        
        if self.left is not None:
            if self.left.mode == b'LEAF' or self.left.feature_id != self.feature_id:
                samples += self.left.samples
            else:
                samples += self.left.same_feature_branch_samples()
        if self.right is not None:
            if self.right.mode == b'LEAF' or self.right.feature_id != self.feature_id:
                samples += self.right.samples
            else:
                samples += self.right.same_feature_branch_samples()
        
        return samples

    # 重置权重
    def replace_samples(self) -> int:
        if self.mode == b'LEAF':
            self.samples = 1
            return self.samples

        self.samples = self.left.replace_samples() + self.right.replace_samples()
        return self.samples
    
    def update_samples(self) -> int:
        if self.mode == b'LEAF':
            return self.samples

        self.samples = self.left.update_samples() + self.right.update_samples()
        return self.samples

    def get_samples_list(self, samples_list: List[int]):
        samples_list.append(self.samples)
        if self.mode != b'LEAF':
            self.left.get_samples_list(samples_list)
            self.right.get_samples_list(samples_list)

    def check_samples(self):
        if self.mode != b'LEAF':
            legal = (self.left.samples + self.right.samples == self.samples)
            if not legal:
                raise ValueError(f'samples not match: {self.left.samples} + {self.right.samples} != {self.samples}')
            self.left.check_samples() 
            self.right.check_samples()

    def max_depth_to_leaf(self) -> int:
        if self.mode == b'LEAF':
            return 0

        return 1 + max(self.left.max_depth_to_leaf(), self.right.max_depth_to_leaf())
    
    def tosql_v1(self, features: List[str]) -> str:
        sql = ''
        if self.mode == b'LEAF':
            sql += f'{self.target_weight:.6f}'
        else:
            if self.left.samples > self.right.samples:
                sql += f'CASE WHEN {features[self.feature_id]} <= {self.value:.6f} THEN {self.left.tosql_v1(features)} ELSE {self.right.tosql_v1(features)} END'
            else:
                sql += f'CASE WHEN {features[self.feature_id]} > {self.value:.6f} THEN {self.right.tosql_v1(features)} ELSE {self.left.tosql_v1(features)} END'
        return sql

    def tosql(self, features: List[str]) -> str:
        if self.mode == b'LEAF':
            return f'{self.target_weight}'

        sql = ''
        if self.mode != b'LEAF':
            sql_l = self.left.tosql(features)
            if sql_l == '1':
                sql = f'{features[self.feature_id]} <= {self.value:.6f}'
            elif sql_l not in ['', '0']:
                sql = f'{features[self.feature_id]} <= {self.value:.6f} AND ({sql_l})'

            sql_r = self.right.tosql(features)
            if sql_r == '1':
                if sql != '':
                    sql = f'({sql}) OR {features[self.feature_id]} > {self.value:.6f}'
                else:
                    sql = f'{features[self.feature_id]} > {self.value:.6f}'
            elif sql_r not in ['', '0']:
                if sql != '':
                    sql = f'({sql}) OR ({features[self.feature_id]} > {self.value:.6f} AND ({sql_r}))'
                else:
                    sql = f'{features[self.feature_id]} > {self.value:.6f} AND ({sql_r})'

            return sql

    def toEchartsJSON(self) -> dict:
        if self.mode == b'LEAF':
            return {
                'name': f'{self.target_weight:.3f}',
                'collapsed': False
            }
        
        return {
            'name': f'x{self.feature_id} <= {self.value:.3f}',
            'collapsed': False,
            'children': [
                self.left.toEchartsJSON(),
                self.right.toEchartsJSON()
            ]
        }

class TreeEnsembleRegressor:
    def __init__(self):
        self.n_targets: int = 1
        self.nodes_falsenodeids: List[int] = []
        self.nodes_featureids: List[int] = []
        self.nodes_hitrates: List[float] = []
        self.nodes_missing_value_tracks_true: List[int] = []
        self.nodes_modes: List[bytes] = []
        self.nodes_nodeids: List[int] = []
        self.nodes_treeids: List[int] = []
        self.nodes_truenodeids: List[int] = []
        self.nodes_values: List[float] = []
        self.post_transform: bytes = b'NONE'
        self.target_ids: List[int] = []
        self.target_nodeids: List[int] = []
        self.target_treeids: List[int] = []
        self.target_weights: List[float] = []

    def to_model(self, input_model: onnx.ModelProto) -> onnx.ModelProto:
        # node
        node = helper.make_node(
            op_type='TreeEnsembleRegressor',
            inputs=[input_model.graph.input[0].name],
            outputs=[input_model.graph.output[0].name],
            name='TreeEnsembleRegressor',
            domain='ai.onnx.ml',
            # attributes
            n_targets=self.n_targets,
            nodes_falsenodeids=self.nodes_falsenodeids,
            nodes_featureids=self.nodes_featureids,
            nodes_hitrates=self.nodes_hitrates,
            nodes_missing_value_tracks_true=self.nodes_missing_value_tracks_true,
            nodes_modes=self.nodes_modes,
            nodes_nodeids=self.nodes_nodeids,
            nodes_treeids=self.nodes_treeids,
            nodes_truenodeids=self.nodes_truenodeids,
            nodes_values=self.nodes_values,
            post_transform=self.post_transform,
            target_ids=self.target_ids,
            target_nodeids=self.target_nodeids,
            target_treeids=self.target_treeids,
            target_weights=self.target_weights
        )

        # graph
        graph = helper.make_graph(
            nodes=[node],
            name=input_model.graph.name,
            initializer=[],
            inputs=input_model.graph.input,
            outputs=input_model.graph.output,
        )

        # model
        output_model = helper.make_model(
            graph=graph,
            opset_imports=input_model.opset_import,
        )
        output_model.ir_version = input_model.ir_version

        onnx.checker.check_model(output_model)

        return output_model
    
    @staticmethod
    def from_trees(roots: List[Node]) -> 'TreeEnsembleRegressor':
        regressors = [TreeEnsembleRegressor.from_tree(root, tree_no) for tree_no, root in enumerate(roots)]
        regressor = TreeEnsembleRegressor()

        for r in regressors:
            regressor.nodes_falsenodeids.extend(r.nodes_falsenodeids)
            regressor.nodes_featureids.extend(r.nodes_featureids)
            regressor.nodes_hitrates.extend(r.nodes_hitrates)
            regressor.nodes_missing_value_tracks_true.extend(r.nodes_missing_value_tracks_true)
            regressor.nodes_modes.extend(r.nodes_modes)
            regressor.nodes_nodeids.extend(r.nodes_nodeids)
            regressor.nodes_treeids.extend(r.nodes_treeids)
            regressor.nodes_truenodeids.extend(r.nodes_truenodeids)
            regressor.nodes_values.extend(r.nodes_values)
            regressor.target_ids.extend(r.target_ids)
            regressor.target_nodeids.extend(r.target_nodeids)
            regressor.target_treeids.extend(r.target_treeids)
            regressor.target_weights.extend(r.target_weights)

        return regressor

    @staticmethod
    def from_tree(root: 'Node', tree_no: int = 0) -> 'TreeEnsembleRegressor':
        regressor = TreeEnsembleRegressor()
        TreeEnsembleRegressor.from_tree_internal(regressor, root, tree_no)
        
        id_map = {old_id: i for i, old_id in enumerate(regressor.nodes_nodeids)}
        # print(id_map)
        is_leaf = [mode == b'LEAF' for mode in regressor.nodes_modes]
        regressor.nodes_falsenodeids = [(0 if is_leaf[i] else id_map[id]) for i, id in enumerate(regressor.nodes_falsenodeids)]
        regressor.nodes_truenodeids = [(0 if is_leaf[i] else id_map[id]) for i, id in enumerate(regressor.nodes_truenodeids)]
        regressor.nodes_nodeids = [id_map[id] for id in regressor.nodes_nodeids]
        regressor.target_nodeids = [id_map[id] for id in regressor.target_nodeids]
        
        return regressor

    @staticmethod
    def from_tree_internal(regressor: 'TreeEnsembleRegressor', node: 'Node', tree_no: int = 0):
        is_leaf = node.mode == b'LEAF'

        regressor.nodes_falsenodeids.append(node.right.id if not is_leaf else 0)
        regressor.nodes_featureids.append(node.feature_id)
        regressor.nodes_hitrates.append(float(node.samples))
        regressor.nodes_missing_value_tracks_true.append(0)
        regressor.nodes_modes.append(node.mode)
        regressor.nodes_nodeids.append(node.id)
        regressor.nodes_treeids.append(tree_no)
        regressor.nodes_truenodeids.append(node.left.id if not is_leaf else 0)
        regressor.nodes_values.append(node.value)
        
        if is_leaf:
            regressor.target_ids.append(0)
            regressor.target_nodeids.append(node.id)
            regressor.target_treeids.append(tree_no)
            regressor.target_weights.append(node.target_weight)

        if not is_leaf:
            TreeEnsembleRegressor.from_tree_internal(regressor, node.left, tree_no)
            TreeEnsembleRegressor.from_tree_internal(regressor, node.right, tree_no)

def get_target_tree_intervals(onnx_model) -> List[Tuple[int, int]]:
    target_tree_roots: List[int] = []
    # target_treeids is ordered
    target_treeids = get_attribute(onnx_model, 'target_treeids').ints
    next_tree_id = 0
    for i, tree_id in enumerate(target_treeids):
        if tree_id == next_tree_id:
            next_tree_id += 1
            target_tree_roots.append(i)

    target_tree_intervals: List[Tuple[int, int]] = []
    for i, root in enumerate(target_tree_roots):
        if i == len(target_tree_roots) - 1:
            end = len(target_treeids)
        else:
            end = target_tree_roots[i + 1]
        target_tree_intervals.append((root, end))
    return target_tree_intervals

def get_tree_intervals(onnx_model) -> List[Tuple[int, int]]:
    tree_roots: List[int] = []
    # nodes_treeids is ordered
    nodes_treeids = get_attribute(onnx_model, 'nodes_treeids').ints
    next_tree_id = 0
    for i, tree_id in enumerate(nodes_treeids):
        if tree_id == next_tree_id:
            next_tree_id += 1
            tree_roots.append(i)

    tree_intervals: List[Tuple[int, int]] = []
    for i, root in enumerate(tree_roots):
        if i == len(tree_roots) - 1:
            end = len(nodes_treeids)
        else:
            end = tree_roots[i + 1]
        tree_intervals.append((root, end))
    return tree_intervals

def model2trees(input_model, samples_list: 'List[int] | None') -> 'List[Node]':
    tree_intervals = get_tree_intervals(input_model)
    target_tree_intervals = get_target_tree_intervals(input_model)
    trees = []
    for tree_no, tree_interval in enumerate(tree_intervals):
        root = model2tree(input_model, samples_list, 0, None, tree_interval, target_tree_intervals[tree_no])
        trees.append(root)
    return trees

def model2tree(input_model, samples_list: 'List[int] | None', node_id, parent: 'Node | None', 
               tree_interval: 'Tuple[int, int] | None' = None, target_tree_interval: 'Tuple[int, int] | None' = None) -> 'Node':
    if tree_interval is None:
        tree_interval = (0, len(get_attribute(input_model, 'nodes_treeids').ints))
    tree_start, tree_end = tree_interval

    if target_tree_interval is None:
        target_tree_interval = (0, len(get_attribute(input_model, 'target_treeids').ints))
    target_tree_start, target_tree_end = target_tree_interval

    # input model attributes
    # # n_targets
    input_n_targets = get_attribute(input_model, 'n_targets').i
    # # nodes_falsenodeids: 右侧分支
    input_nodes_falsenodeids = get_attribute(input_model, 'nodes_falsenodeids').ints[tree_start:tree_end]
    # # nodes_featureids: 特征id
    input_nodes_featureids = get_attribute(input_model, 'nodes_featureids').ints[tree_start:tree_end]
    # # nodes_hitrates
    input_nodes_hitrates = get_attribute(input_model, 'nodes_hitrates').floats[tree_start:tree_end]
    # # nodes_missing_value_tracks_true
    input_nodes_missing_value_tracks_true = get_attribute(input_model, 'nodes_missing_value_tracks_true').ints[tree_start:tree_end]
    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    input_node_modes = get_attribute(input_model, 'nodes_modes').strings[tree_start:tree_end]
    # # nodes_nodeids
    input_nodes_nodeids = get_attribute(input_model, 'nodes_nodeids').ints[tree_start:tree_end]
    # # nodes_treeids
    input_nodes_treeids = get_attribute(input_model, 'nodes_treeids').ints[tree_start:tree_end]
    # # nodes_truenodeids: 左侧分支
    input_nodes_truenodeids = get_attribute(input_model, 'nodes_truenodeids').ints[tree_start:tree_end]
    # # nodes_values: 阈值，叶子节点的值为0
    input_nodes_values = get_attribute(input_model, 'nodes_values').floats[tree_start:tree_end]
    # # post_transform
    input_post_transform = get_attribute(input_model, 'post_transform').s
    # # target_ids
    input_target_ids = get_attribute(input_model, 'target_ids').ints[target_tree_start:target_tree_end]
    # # target_nodeids: 叶子节点的id
    input_target_nodeids = get_attribute(input_model, 'target_nodeids').ints[target_tree_start:target_tree_end]
    # # target_treeids
    input_target_treeids = get_attribute(input_model, 'target_treeids').ints[target_tree_start:target_tree_end]
    # # target_weights: 叶子节点的权重，即预测值
    input_target_weights = get_attribute(input_model, 'target_weights').floats[target_tree_start:target_tree_end]

    # node_id -> target_id
    input_target_nodeid_map = {node_id: i for i, node_id in enumerate(input_target_nodeids)}

    id = node_id
    feature_id = input_nodes_featureids[id]
    mode = input_node_modes[id]
    value = input_nodes_values[id]
    target_id = input_target_nodeid_map.get(id, None)
    target_weight = input_target_weights[target_id] if target_id is not None else None
    samples = int(input_nodes_hitrates[id])
    
    # only for debug
    if samples_list is not None:
        tree_samples_list = samples_list[tree_start:tree_end]
        if samples != tree_samples_list[id]:
            raise ValueError(f'samples not match: {samples} != {tree_samples_list[id]}')
    
    node = Node(
        id=id,
        feature_id=feature_id,
        mode=mode,
        value=value,
        target_id=target_id,
        target_weight=target_weight,
        samples=samples
    )
    node.parent = parent
    
    if mode != b'LEAF':
        left_node_id = input_nodes_truenodeids[id]
        left_node = model2tree(input_model, samples_list, left_node_id, node, tree_interval, target_tree_interval)
        node.left = left_node

        right_node_id = input_nodes_falsenodeids[id]
        right_node = model2tree(input_model, samples_list, right_node_id, node, tree_interval, target_tree_interval)
        node.right = right_node

    return node

def clf2reg(input_model: onnx.ModelProto) -> onnx.ModelProto:
    # input model attributes
    # # class_ids: 叶子节点权重对应的类别id
    input_class_ids = get_attribute(input_model, 'class_ids').ints
    # # class_nodeids: 叶子节点权重对应的节点id
    input_class_nodeids = get_attribute(input_model, 'class_nodeids').ints
    # # class_treeids: 叶子节点权重对应的树id
    input_class_treeids = get_attribute(input_model, 'class_treeids').ints
    # # class_weights: 叶子节点权重，即预测值
    input_class_weights = get_attribute(input_model, 'class_weights').floats
    # # classlabels_int64s: 类别id
    input_classlabels_int64s = get_attribute(input_model, 'classlabels_int64s').ints
    # # nodes_falsenodeids: 右侧分支
    input_nodes_falsenodeids = get_attribute(input_model, 'nodes_falsenodeids').ints
    # # nodes_featureids: 特征id
    input_nodes_featureids = get_attribute(input_model, 'nodes_featureids').ints
    # # nodes_hitrates
    input_nodes_hitrates = get_attribute(input_model, 'nodes_hitrates').floats
    # # nodes_missing_value_tracks_true
    input_nodes_missing_value_tracks_true = get_attribute(input_model, 'nodes_missing_value_tracks_true').ints
    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    input_nodes_modes = get_attribute(input_model, 'nodes_modes').strings
    # # nodes_nodeids
    input_nodes_nodeids = get_attribute(input_model, 'nodes_nodeids').ints
    # # nodes_treeids
    input_nodes_treeids = get_attribute(input_model, 'nodes_treeids').ints
    # # nodes_truenodeids: 左侧分支
    input_nodes_truenodeids = get_attribute(input_model, 'nodes_truenodeids').ints
    # # nodes_values: 阈值，叶子节点的值为0
    input_nodes_values = get_attribute(input_model, 'nodes_values').floats
    # # post_transform
    input_post_transform = get_attribute(input_model, 'post_transform').s

    n_trees = len(set(input_class_treeids))

    # output model attributes
    # # n_targets
    n_targets = 1

    # # nodes_falsenodeids: 右侧分支
    nodes_falsenodeids = input_nodes_falsenodeids

    # # nodes_featureids: 特征id
    nodes_featureids = input_nodes_featureids

    # # nodes_hitrates
    nodes_hitrates = input_nodes_hitrates

    # # nodes_missing_value_tracks_true
    nodes_missing_value_tracks_true = input_nodes_missing_value_tracks_true

    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    nodes_modes = input_nodes_modes

    # # nodes_nodeids
    nodes_nodeids = input_nodes_nodeids

    # # nodes_treeids
    nodes_treeids = input_nodes_treeids

    # # nodes_truenodeids: 左侧分支
    nodes_truenodeids = input_nodes_truenodeids

    # # nodes_values: 阈值，叶子节点的值为0
    nodes_values = input_nodes_values

    # # post_transform
    post_transform = input_post_transform

    stride = len(input_classlabels_int64s)
    if stride == 2:
        stride = 1

    n_leaf = len(input_class_weights) // stride

    # # target_ids
    target_ids = []
    for i in range(n_leaf):
        target_ids.append(input_class_ids[i * stride])

    # # target_nodeids: 叶子节点的id
    target_nodeids = []
    for i in range(n_leaf):
        target_nodeids.append(input_class_nodeids[i * stride])

    # # target_treeids
    target_treeids = []
    for i in range(n_leaf):
        target_treeids.append(input_class_treeids[i * stride])
    
    # # target_weights: 叶子节点的权重，即预测值
    target_weights = []
    if stride == 1:
        # binary mode: only store positive class weight
        target_weights = [1.0 if w > 0.5 / n_trees else 0.0 for w in input_class_weights]
    else:
        for i in range(n_leaf):
            targets = input_class_weights[i * stride: (i + 1) * stride]
            target_weights.append(float(np.argmax(targets)))
    
    # node
    node = helper.make_node(
        op_type='TreeEnsembleRegressor',
        inputs=[input_model.graph.input[0].name],
        outputs=[input_model.graph.output[0].name],
        name='TreeEnsembleRegressor',
        domain='ai.onnx.ml',
        # attributes
        n_targets=n_targets,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_featureids=nodes_featureids,
        nodes_hitrates=nodes_hitrates,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        nodes_modes=nodes_modes,
        nodes_nodeids=nodes_nodeids,
        nodes_treeids=nodes_treeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_values=nodes_values,
        post_transform=post_transform,
        target_ids=target_ids,
        target_nodeids=target_nodeids,
        target_treeids=target_treeids,
        target_weights=target_weights
    )

    # 替换
    # 1. 删除
    # 2. 新增
    # 3. 设置图的新输出
    
    # target_node_name = 'TreeEnsembleClassifier'
    # target_node = None

    # # 查找目标节点
    # for node in input_model.graph.node:
    #     if node.name == target_node_name:
    #         target_node = node
    #         break
    
    
    output = helper.make_tensor_value_info(
        name=input_model.graph.output[0].name,
        elem_type=onnx.TensorProto.FLOAT,
        shape=[None, 1],
    )
    graph = helper.make_graph(
        nodes=[node],
        name=input_model.graph.name,
        initializer=[],
        inputs=input_model.graph.input,
        outputs=[output],
    )

    # model
    output_model = helper.make_model(
        graph=graph,
        opset_imports=input_model.opset_import,
    )
    output_model.ir_version = input_model.ir_version

    onnx.checker.check_model(output_model)
    
    input_model.graph.node.remove()
    input_model.graph.node.append()

    return output_model