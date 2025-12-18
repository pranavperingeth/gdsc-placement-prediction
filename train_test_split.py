import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("placement_data.csv")

X = df.drop("placed", axis=1)
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)
