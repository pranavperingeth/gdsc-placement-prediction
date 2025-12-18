import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("placement_data.csv")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())
sns.countplot(x="placed", data=df)
plt.title("Placement Distribution")
plt.xlabel("Placed (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
sns.boxplot(x="placed", y="cgpa", data=df)

plt.title("CGPA vs Placement")
plt.show()
sns.boxplot(x="placed", y="internships", data=df)

plt.title("Internships vs Placement")
sns.countplot(x="branch", hue="placed", data=df)
plt.title("Branch vs Placement")

plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
