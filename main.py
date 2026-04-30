import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("student.csv")

print("Dataset Preview:")
print(df.head())

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop("final_grade", axis=1)
y = df["final_grade"]

# -----------------------------
# FIX 1: Convert ALL text columns in X (male, female, etc.)
# -----------------------------
X = pd.get_dummies(X)

# -----------------------------
# FIX 2: Encode target (grades like a,b,c,d,e)
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(y)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model (classification)
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save model + encoder
# -----------------------------
joblib.dump(model, "student_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel saved successfully!")