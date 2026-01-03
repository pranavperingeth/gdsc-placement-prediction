# Placement Prediction using Logistic Regression (From Scratch)

This project implements a **placement prediction system from scratch** using only **Python, NumPy, and Pandas**, without relying on high-level machine learning libraries such as scikit-learn.

The goal is to understand how **Logistic Regression actually works internally**, including feature scaling, probability computation, loss calculation, and gradient descent.

---

## 📌 Problem Statement

Predict whether a student will be **placed or not** based on academic and skill-related features such as:
- CGPA
- Number of internships
- Number of projects
- Aptitude test score

---

## 🧠 Approach

The model is built entirely using matrix operations and follows these steps:

1. Load and preprocess the dataset
2. Encode categorical labels manually
3. Perform **feature standardization** using mean and standard deviation
4. Implement Logistic Regression using:
   - Sigmoid function
   - Log loss (binary cross-entropy)
   - Gradient descent
5. Train the model iteratively
6. Evaluate accuracy manually
7. Predict placement probability for user-provided input

---

## 🛠 Technologies Used
- Python
- NumPy
- Pandas

---

## 📂 File Structure

