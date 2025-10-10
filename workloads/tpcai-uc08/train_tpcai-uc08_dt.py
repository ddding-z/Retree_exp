import numpy as np
import onnxoptimizer
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import onnx
import datetime
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
import argparse

import sys
import os

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import value_distribution

""" 
tpcai-uc08:
多表分类任务

mse: 0.15457174146049696
rmsle: 0.23555262282642453
r2: 0.571661716545319

python train_tpcai-uc08_dt.py -td 10
"""


parser = argparse.ArgumentParser()
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "tpcai-uc08"
tree_depth = args.tree_depth
label = "trip_type"

# load data
data_path = f"data/{data_name}-train.csv"
data = pd.read_csv(data_path)

data.columns = data.columns.str.replace(' ', '_')
data.columns = data.columns.str.replace('/', '_')
data.columns = data.columns.str.replace(',', '')
data.columns = data.columns.str.replace('-', '')
data.columns = data.columns.str.replace('&', '')
data.columns = data.columns.str.replace('1HR', 'HR')

# choose feature: 4 numerical
X = data.drop(label, axis=1)
input_columns = list(X.columns)

# define pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", input_columns),
    ]
)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("Classifier", DecisionTreeClassifier(max_depth=tree_depth)),
    ]
)

# define data
y = np.array(data.loc[:, label].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"{classification_report(y_test, y_pred)}")

model = pipeline.named_steps["Classifier"]
depth = model.get_depth()
leaves = model.get_n_leaves()
node_count = model.tree_.node_count
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

model_name = f"{data_name}_d{depth}_l{leaves}_n{node_count}_{now}"
onnx_path = f"model/{model_name}.onnx"

# save model pred distribution
pred = pipeline.predict(X)
value_distribution(pred, model_name)

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
onnx.save_model(model_onnx, onnx_path)

# optimize model
# optimized_model = onnxoptimizer.optimize(model_onnx)
# onnx.save_model(optimized_model, onnx_path)

with open(f"/volumn/Retree_exp/queries/Retree/workloads/workload_models.csv", "a", encoding="utf-8") as f:
    f.write(f"tpcai-uc08,{model_name}\n")
#     # f.write(f"{data_name},{model_name}\n")
