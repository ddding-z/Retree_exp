import pandas as pd

# python flights_expand.py

scale_1G = 140

path1 = "S_routes.csv"
path2 = "R1_airlines.csv"
path3 = "R2_sairports.csv"
path4 = "R3_dairports.csv"

outpath1 = "../data-extension/1G/"
outpath2 = "../data-extension/10G/"
outpath3 = "../data-extension/"

# S_routes = pd.read_csv(path1).iloc[:, 0:4]
# R1_airlines = pd.read_csv(path2)
# R2_sairports = pd.read_csv(path3)
# R3_dairports = pd.read_csv(path4)

# S_routes["airlineid"] = S_routes["airlineid"].str.replace("'", "")
# S_routes["sairportid"] = S_routes["sairportid"].str.replace("'", "")
# S_routes["dairportid"] = S_routes["dairportid"].str.replace("'", "")

# R1_airlines["airlineid"] = R1_airlines["airlineid"].str.replace("'", "")
# R2_sairports["sairportid"] = R2_sairports["sairportid"].str.replace("'", "")
# R3_dairports["dairportid"] = R3_dairports["dairportid"].str.replace("'", "")

# R1_airlines.to_csv(outpath3 + path2, index=False)
# R2_sairports.to_csv(outpath3 + path3, index=False)
# R3_dairports.to_csv(outpath3 + path4, index=False)

# X = S_routes.drop("codeshare", axis=1)
# X_expanded = pd.concat([X] * scale_1G, ignore_index=True)

# data = pd.merge(
#     pd.merge(pd.merge(X_expanded, R1_airlines, how="inner"), R2_sairports, how="inner"),
#     R3_dairports,
#     how="inner",
# )
# data.to_csv("flights_1G.csv", index=False)
# X_expanded.to_csv(outpath1 + path1, index=False)

# expand to 10G
X_expanded_1G = pd.read_csv(outpath1 + path1)
X_expanded_10G = pd.concat([X_expanded_1G] * 10, ignore_index=True)
X_expanded_10G.to_csv(outpath2 + path1, index=False)


