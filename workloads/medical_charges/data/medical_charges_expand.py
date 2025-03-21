import pandas as pd

# python medical_charges_expand.py

scale_1G = 256

path = "medical_charges.csv"

outpath1 = "../data-extension/1G/"
outpath2 = "../data-extension/10G/"

df = pd.read_csv(path)

X = df.drop('AverageTotalPayments', axis=1)
X_expanded_1G = pd.concat([X] * scale_1G, ignore_index=True)
X_expanded_1G.to_csv(outpath1 + path, index=False)

# expand to 10G
# X_expanded_1G = pd.read_csv(outpath1 + path)
# X_expanded_10G = pd.concat([X_expanded_1G] * 10, ignore_index=True)
# X_expanded_10G.to_csv(outpath2 + path, index=False)
