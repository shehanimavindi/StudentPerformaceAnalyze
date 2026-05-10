import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\diluk\OneDrive\Desktop\Projects\StudentPerfomanceAnalyze\student.csv")

# Encode grades for colors
le = LabelEncoder()
df["grade_encoded"] = le.fit_transform(df["final_grade"])

df_sample = df.sample(100, random_state=42)

# Create figure
plt.figure(figsize=(10,6))

# Scatter plot used smaller dots and transparency for clear graph
scatter = plt.scatter(
    df_sample["study_hours"],
    df_sample["attendance_percentage"],
    c=df_sample["grade_encoded"],
    cmap="viridis",
    s=60,          
    alpha=0.5      
)

# Labels
plt.xlabel("Study Hours")
plt.ylabel("Attendance Percentage")
plt.title("Student Performance Scatter Plot")

# Legend
handles, _ = scatter.legend_elements()
plt.legend(handles, le.classes_, title="Grades")

# Grid
plt.grid(True)

plt.show()