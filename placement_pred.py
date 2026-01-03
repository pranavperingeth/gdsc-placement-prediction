import numpy as np
import pandas as pd

# ================================
# 1. LOAD DATA
# ================================

df = pd.read_csv("placement_data.csv")

# Keep only required columns
df = df[[
    "CGPA",
    "Internships",
    "Projects",
    "AptitudeTestScore",
    "PlacementStatus"
]]

# Encode target: Placed -> 1, NotPlaced -> 0
df["PlacementStatus"] = df["PlacementStatus"].map({
    "Placed": 1,
    "NotPlaced": 0
})

# ================================
# 2. FEATURE MATRIX & LABEL
# ================================

X = df[["CGPA", "Internships", "Projects", "AptitudeTestScore"]].values
y = df["PlacementStatus"].values.reshape(-1, 1)

# ================================
# 3. FEATURE SCALING (MANUAL)
# ================================

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X = (X - X_mean) / X_std

# ================================
# 4. INITIALIZE PARAMETERS
# ================================

np.random.seed(0)
W = np.random.randn(X.shape[1], 1)
b = 0

# ================================
# 5. HELPER FUNCTIONS
# ================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, W, b):
    z = np.dot(X, W) + b
    return sigmoid(z)

def compute_loss(y, y_hat):
    m = len(y)
    loss = - (1/m) * np.sum(
        y * np.log(y_hat + 1e-9) +
        (1 - y) * np.log(1 - y_hat + 1e-9)
    )
    return loss

def compute_gradients(X, y, y_hat):
    m = len(y)
    dW = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dW, db

# ================================
# 6. TRAINING LOOP
# ================================

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    y_hat = predict(X, W, b)
    loss = compute_loss(y, y_hat)
    dW, db = compute_gradients(X, y, y_hat)

    W -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ================================
# 7. MODEL EVALUATION
# ================================

y_pred = predict(X, W, b)
y_pred_class = (y_pred >= 0.5).astype(int)

accuracy = np.mean(y_pred_class == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")

# ================================
# 8. USER INPUT PREDICTION
# ================================

def predict_student(cgpa, internships, projects, aptitude):
    user = np.array([[cgpa, internships, projects, aptitude]])
    user = (user - X_mean) / X_std
    probability = predict(user, W, b)[0][0]
    return probability

print("\n--- Student Placement Prediction ---")
cgpa = float(input("Enter CGPA: "))
internships = int(input("Enter number of internships: "))
projects = int(input("Enter number of projects: "))
aptitude = float(input("Enter aptitude test score: "))

prob = predict_student(cgpa, internships, projects, aptitude)

print(f"\nProbability of being placed: {prob * 100:.2f}%")
if prob >= 0.5:
    print("Prediction: Student is likely to be PLACED")
else:
    print("Prediction: Student is likely to be NOT PLACED")

