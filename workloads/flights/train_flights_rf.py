import argparse
import datetime
import joblib
import numpy as np
import onnx.checker
import onnxoptimizer
import pandas as pd
import onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import value_distribution, plot_feature_importances

""" 
flights:
多表分类任务
              precision    recall  f1-score   support

           0       0.80      0.96      0.87       513
           1       0.60      0.19      0.29       153

    accuracy                           0.79       666
   macro avg       0.70      0.58      0.58       666
weighted avg       0.75      0.79      0.74       666

python train_flights_rf.py -tn 100 -td 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tree_num", "-tn", type=int, default=100)
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "flights"
tree_num = args.tree_num
tree_depth = args.tree_depth
label = "codeshare"

path1 = "data/S_routes.csv"
path2 = "data/R1_airlines.csv"
path3 = "data/R2_sairports.csv"
path4 = "data/R3_dairports.csv"

# load data
S_routes = pd.read_csv(path1)
R1_airlines = pd.read_csv(path2)
R2_sairports = pd.read_csv(path3)
R3_dairports = pd.read_csv(path4)

data = pd.merge(
    pd.merge(pd.merge(S_routes, R1_airlines, how="inner"), R2_sairports, how="inner"),
    R3_dairports,
    how="inner",
)
data.dropna(inplace=True)
data[label] = data[label].replace({"f": 0, "t": 1}).astype("int")
# data.head(2048).to_csv(f"data/{data_name}-2048.csv', index=False)
# data.to_csv('data/{data_name}.csv', index=False)

# choose feature: 4 numerical, 13 categorical
numerical = [
    "slatitude",
    "slongitude",
    "dlatitude",
    "dlongitude"
]
categorical = [
    # "acountry",
    "active",
    # "scity",
    # "scountry",
    # "stimezone",
    "sdst",
    # "dcity",
    # "dcountry",
    # "dtimezone",
    "ddst",
]
input_columns = numerical + categorical

# define pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "Classifier",
            RandomForestClassifier(
                n_estimators=tree_num, max_depth=tree_depth, n_jobs=50
            ),
        ),
    ]
)

# define data
X = data.loc[:, input_columns]
y = np.array(data.loc[:, label].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)

# train
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"{classification_report(y_test, y_pred)}")

# define path
model = pipeline.named_steps["Classifier"]
depth = [model.estimators_[i].get_depth() for i in range(tree_num)]
leaves = [model.estimators_[i].get_n_leaves() for i in range(tree_num)]
node_count = [model.estimators_[i].tree_.node_count for i in range(tree_num)]
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

model_name = f"{data_name}_t{tree_num}_d{sum(depth) // tree_num}_l{sum(leaves) // tree_num}_n{sum(node_count) //tree_num}_{now}"
joblib.dump(pipeline, f"model/{model_name}.joblib")
onnx_path = f"model/{model_name}.onnx"
# save model pred distribution
pred = pipeline.predict(X)
value_distribution(pred, model_name)
# plot_feature_importances(model, X.shape[1], model_name)

# convert and save model
type_map = {
    "int64": Int64TensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1]),
}
init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]
model_onnx = convert_sklearn(pipeline, initial_types=init_types)
# model_onnx = convert_sklearn(pipeline, initial_types=init_types,options={id(model): {'zipmap': False}})

# optimize model
optimized_model = onnxoptimizer.optimize(model_onnx)
onnx.save_model(optimized_model, onnx_path)
# onnx.checker.check_model(model_onnx)
# onnx.save_model(model_onnx, onnx_path)

with open(f"/volumn/Retree_exp/queries/Retree/workloads/workload_models.csv", "a", encoding="utf-8") as f:
    f.write(f"{data_name},{model_name}\n")
