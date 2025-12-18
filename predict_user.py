import pickle
import numpy as np

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("\n--- Placement Prediction System ---")

cgpa = float(input("Enter CGPA: "))
branch = int(input("Enter Branch (0=CSE, 1=ECE, 2=ME): "))
internships = int(input("Enter number of internships: "))

user_data = np.array([[cgpa, branch, internships]])
user_data = scaler.transform(user_data)

prediction = model.predict(user_data)[0]
probability = model.predict_proba(user_data)[0][1] * 100

print(f"\nProbability of getting placed: {probability:.2f}%")

if prediction == 1:
    print("Final Result: Student is likely to be PLACED ✅")
else:
    print("Final Result: Student is NOT likely to be placed ❌")


