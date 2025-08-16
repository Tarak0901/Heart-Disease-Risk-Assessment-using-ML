import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import joblib

# Load the Cleveland Heart Disease dataset
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, names=column_names, na_values="?")
df.dropna(inplace=True)
df = df.astype(float)

# Convert target to binary (0 = no disease, 1 = disease)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM with Grid Search
params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
grid = GridSearchCV(SVC(probability=True), params, cv=5)
grid.fit(X_train_scaled, y_train)

# Save the scaler and model
joblib.dump(scaler, "scaler.pkl")
joblib.dump(grid, "svm_model.pkl")

print("scaler.pkl and svm_model.pkl have been saved successfully!")