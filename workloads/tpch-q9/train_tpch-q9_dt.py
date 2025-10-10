import numpy as np
import onnxoptimizer
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
import onnx
import datetime
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
import argparse

import sys
import os

from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import plot_hist, plot_value_distribution, percentile_values

""" 
tpch-q9:
多表回归任务

mse: 2205876.8447108986
r2: 0.993009605279969

python train_tpch-q9_dt.py -td 10
"""


parser = argparse.ArgumentParser()
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "tpch-q9"
tree_depth = args.tree_depth
label = "amount"

# load data
data_path = f"data/{data_name}.csv"
data = pd.read_csv(data_path)
# data.head(2048).to_csv(f"data/{data_name}-2048.csv", index=False)

# choose feature: 4 numerical
numerical = [
    "l_extendedprice",
    "l_discount",
    "ps_supplycost",
    "l_quantity",
]
input_columns = numerical

# define pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", input_columns),
    ]
)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("Regressor", DecisionTreeRegressor(max_depth=tree_depth)),
    ]
)

# define data
X = data.loc[:, input_columns]
y = np.array(data.loc[:, label].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"mse: {mean_squared_error(y_test, y_pred)}")
# print(f"rmsle: {np.sqrt(mean_squared_log_error(y_test, y_pred))}")
print(f"r2: {r2_score(y_test, y_pred)}")

model = pipeline.named_steps["Regressor"]
depth = model.get_depth()
leaves = model.get_n_leaves()
node_count = model.tree_.node_count
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

model_name = f"{data_name}_d{depth}_l{leaves}_n{node_count}_{now}"
onnx_path = f"model/{model_name}.onnx"

# save model pred distribution
pred = pipeline.predict(X)
plot_value_distribution(pred, model_name)
percentile_values(pred, data_name, model_name)

# convert and save model
type_map = {
    "int64": Int64TensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1]),
}
init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]
model_onnx = convert_sklearn(pipeline, initial_types=init_types)

# optimize model
optimized_model = onnxoptimizer.optimize(model_onnx)
onnx.save_model(optimized_model, onnx_path)

# with open(f"/volumn/Retree_exp/queries/Retree/workloads/workload_models.csv", "a", encoding="utf-8") as f:
#     f.write(f"{data_name},{model_name}\n")
