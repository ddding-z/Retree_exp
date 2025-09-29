import argparse
import datetime
import numpy as np
import onnxoptimizer
import pandas as pd
import onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
import joblib

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import percentile_values, plot_feature_importances, plot_value_distribution, value_distribution

def wmae_test(test, pred): # WMAE for test 
    weights = X_test['IsHoliday'].apply(lambda is_holiday:5 if is_holiday else 1)
    error = np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)
    return error

""" 
walmart-sales: walmart kaggle version
多表回归任务
              precision    recall  f1-score   support

           1       0.00      0.00      0.00        12
           2       0.66      0.65      0.66       661
           3       0.55      0.63      0.59       968
           4       0.95      0.02      0.05       747
           5       0.41      0.83      0.54       943
           6       0.90      0.54      0.68       836
           7       1.00      0.06      0.12        49

    accuracy                           0.55      4216
   macro avg       0.64      0.39      0.38      4216
weighted avg       0.68      0.55      0.50      4216

python train_walmart_sales_rf.py -tn 100 -td 10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--tree_num", "-tn", type=int, default=100)
parser.add_argument("--tree_depth", "-td", type=int, default=10)
args = parser.parse_args()

data_name = "walmart_sales"
tree_num = args.tree_num
tree_depth = args.tree_depth
label = "Weekly_Sales"

path1 = "data/train.csv"
path2 = "data/features.csv"
path3 = "data/stores.csv"

sales = pd.read_csv(path1)  # sales : train
features = pd.read_csv(path2)
stores = pd.read_csv(path3)

# data preprocessing
data = pd.merge(
    pd.merge(sales, features, on=["Store", "Date", "IsHoliday"], how="inner"),
    stores,
    on="Store",
    how="inner",
)
data.dropna(inplace=True)

data.loc[
    (data["Date"] == "2010-02-12")
    | (data["Date"] == "2011-02-11")
    | (data["Date"] == "2012-02-10"),
    "Super_Bowl",
] = True
data.loc[
    (data["Date"] != "2010-02-12")
    & (data["Date"] != "2011-02-11")
    & (data["Date"] != "2012-02-10"),
    "Super_Bowl",
] = False

data.loc[
    (data["Date"] == "2010-09-10")
    | (data["Date"] == "2011-09-09")
    | (data["Date"] == "2012-09-07"),
    "Labor_Day",
] = True
data.loc[
    (data["Date"] != "2010-09-10")
    & (data["Date"] != "2011-09-09")
    & (data["Date"] != "2012-09-07"),
    "Labor_Day",
] = False

data.loc[
    (data["Date"] == "2010-11-26") | (data["Date"] == "2011-11-25"), "Thanksgiving"
] = True
data.loc[
    (data["Date"] != "2010-11-26") & (data["Date"] != "2011-11-25"), "Thanksgiving"
] = False

data.loc[
    (data["Date"] == "2010-12-31") | (data["Date"] == "2011-12-30"), "Christmas"
] = True
data.loc[
    (data["Date"] != "2010-12-31") & (data["Date"] != "2011-12-30"), "Christmas"
] = False

# convert to datetime
data["Date"] = pd.to_datetime(data["Date"])
data["week"] = data["Date"].dt.isocalendar().week
data["month"] = data["Date"].dt.month
data["year"] = data["Date"].dt.year

type_group = {"A": 1, "B": 2, "C": 3}  # changing A,B,C to 1-2-3
data["Type"] = data["Type"].replace(type_group)
data["Super_Bowl"] = data["Super_Bowl"].astype(bool).astype(int)
data["Thanksgiving"] = data["Thanksgiving"].astype(bool).astype(int)
data["Labor_Day"] = data["Labor_Day"].astype(bool).astype(int)
data["Christmas"] = data["Christmas"].astype(bool).astype(int)
data["IsHoliday"] = data["IsHoliday"].astype(bool).astype(int)

# data.head(2048).to_csv(
#     f"data/{data_name}-2048.csv", index=False
# )
# data.to_csv(f"data/{data_name}.csv", index=False)


# 2 categorical, 10 numerical
numerical = [
    "Store",
    "Dept",
    # "Date",
    "IsHoliday",
    # "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    # "MarkDown4",
    # "MarkDown5",
    # "CPI",
    # "Unemployment",
    "Type",
    "Size",
    "Super_Bowl",
    "Labor_Day",
    "Thanksgiving",
    "Christmas",
    "week",
    "month",
    "year"
]

input_columns = numerical

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical),
        # (
        #     "cat",
        #     OneHotEncoder(handle_unknown="ignore"),
        #     categorical,
        # ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "Regressor",
            RandomForestRegressor(
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
print(f"score: {pipeline.score(X_test, y_test)}")
print(f"wmae: {wmae_test(y_test, y_pred)}")

# define path
model = pipeline.named_steps["Regressor"]
depth = [model.estimators_[i].get_depth() for i in range(tree_num)]
leaves = [model.estimators_[i].get_n_leaves() for i in range(tree_num)]
node_count = [model.estimators_[i].tree_.node_count for i in range(tree_num)]
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

model_name = f"{data_name}_t{tree_num}_d{sum(depth) // tree_num}_l{sum(leaves) // tree_num}_n{sum(node_count) //tree_num}_{now}"
joblib.dump(pipeline, f"model/{model_name}.joblib")
onnx_path = f"model/{model_name}.onnx"

# save model pred distribution
pred = pipeline.predict(X)
plot_value_distribution(pred, model_name)
percentile_values(pred, data_name, model_name)

# convert and save model
type_map = {
    # "int64": Int64TensorType([None, 1]),
    "int64": FloatTensorType([None, 1]),
    "UInt32": FloatTensorType([None, 1]),
    "int32": FloatTensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1]),
}
init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]
model_onnx = convert_sklearn(pipeline, initial_types=init_types)

# optimize model
optimized_model = onnxoptimizer.optimize(model_onnx)
onnx.save_model(optimized_model, onnx_path)

with open(f"/volumn/Retree_exp/queries/Retree/workloads/workload_models.csv", "a", encoding="utf-8") as f:
    f.write(f"{data_name},{model_name}\n")
