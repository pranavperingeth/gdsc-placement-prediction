import pandas as pd

df = pd.read_csv("placement_data.csv")

X = df.drop("placed", axis=1)
y = df["placed"]

print("FEATURES (X):")
print(X.head())

print("\nTARGET (y):")
print(y.head())
