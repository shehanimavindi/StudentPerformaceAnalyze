import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\diluk\OneDrive\Desktop\Projects\StudentPerfomanceAnalyze\student.csv")

# Convert grades into numbers
le = LabelEncoder()
df["grade_numeric"] = le.fit_transform(df["final_grade"])

# Take sample
df_sample = df.sample(80, random_state=42)

# X and y
X = df_sample[["study_hours"]]
y = df_sample["grade_numeric"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Sort data for smooth line
sorted_data = df_sample.sort_values(by="study_hours")

# Predictions
predictions = model.predict(sorted_data[["study_hours"]])

# Create graph
plt.figure(figsize=(10,6))

# Scatter plot
plt.scatter(
    sorted_data["study_hours"],
    sorted_data["grade_numeric"],
    alpha=0.6,
    s=60,
    label="Students"
)

# Regression line
plt.plot(
    sorted_data["study_hours"],
    predictions,
    linewidth=3,
    label="Regression Line"
)

# Labels
plt.xlabel("Study Hours")
plt.ylabel("Grade Level")
plt.title("Linear Regression: Study Hours vs Final Grade")

# Replace numbers with grade labels
plt.yticks(
    range(len(le.classes_)),
    le.classes_
)

plt.legend()
plt.grid(True)

# Show graph
plt.show()