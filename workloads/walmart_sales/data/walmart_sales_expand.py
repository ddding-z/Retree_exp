import pandas as pd

# python walmart_expand.py

scale_1G = 105

path1 = "test.csv"
path2 = "features.csv"
path3 = "stores.csv"

outpath1 = "../data-extension/1G/"
outpath2 = "../data-extension/10G/"
outpath3 = "../data-extension/"

# sales = pd.read_csv(path1)
# features = pd.read_csv(path2)
# stores = pd.read_csv(path3)

# features.to_csv(outpath3 + path2, index=False)
# stores.to_csv(outpath3 + path3, index=False)

# X = sales
# X_expanded = pd.concat([X] * scale_1G, ignore_index=True)

# data = pd.merge(
#     pd.merge(X_expanded, features, on=["Store", "Date", "IsHoliday"], how="inner"),
#     stores,
#     on="Store",
#     how="inner",
# )

# data.to_csv('walmarts_sales_1G.csv', index=False)

# X_expanded.to_csv(outpath1 + path1, index=False)

# expand to 10G
X_expanded_1G = pd.read_csv(outpath1 + path1)
X_expanded_10G = pd.concat([X_expanded_1G] * 10, ignore_index=True)
X_expanded_10G.to_csv(outpath2 + path1, index=False)
