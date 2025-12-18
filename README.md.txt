Placement Prediction System (GDSC AI/ML Inductions)
Overview

This project predicts whether a student is likely to get placed based on key academic and experiential features using Logistic Regression.
The goal is to demonstrate a complete beginner-level ML pipeline: data analysis, model training, evaluation, and user prediction.

Features Used

CGPA

Branch (encoded)

0 → CSE

1 → ECE

2 → ME

Number of Internships

Dataset

A realistic synthetic dataset (placement_data.csv) was created based on common placement trends:

Higher CGPA increases placement probability

Internships significantly improve chances

CSE generally has higher placement rates compared to other branches

Exploratory Data Analysis (EDA)

EDA was performed to understand trends and relationships:

Placement distribution

CGPA vs Placement

Internships vs Placement

Branch-wise placement trends

Correlation heatmap

Key insight: CGPA and internships show strong positive correlation with placement.

Model Used

Logistic Regression

Reason: Suitable for binary classification (Placed / Not Placed)

Model Evaluation

Train–test split: 80% / 20%

Feature scaling: StandardScaler

Evaluation metrics:

Accuracy

Confusion Matrix

The model achieved accuracy greater than 60%, satisfying the induction requirement.

User Prediction

After training, the model is saved and reused for inference.
Users can input:

CGPA

Branch

Number of internships

The system outputs:

Probability of placement (%)

Final placement prediction

Project Structure
gdsc-placement-prediction/
│
├── placement_data.csv
├── eda.py
├── train_and_save.py
├── evaluate_model.py
├── predict_user.py
├── model.pkl
├── scaler.pkl
└── README.md

How to Run

Install dependencies:

pip install pandas scikit-learn numpy seaborn matplotlib


Train and save the model:

python train_and_save.py


Evaluate the model:

python evaluate_model.py


Predict placement (user input):

python predict_user.py

Conclusion

This project demonstrates a complete beginner-friendly ML workflow with clear feature selection, evaluation, and user interaction. It emphasizes understanding and correctness over complexity.