import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, names=column_names, na_values="?")

# Drop missing values
df.dropna(inplace=True)
df = df.astype(float)

# Convert target to binary
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================
# Model: SVM
# ================
svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
svm_grid = GridSearchCV(SVC(class_weight="balanced"), svm_params, cv=5)
svm_grid.fit(X_train, y_train)
svm_pred = svm_grid.predict(X_test)

# ================
# Model: Decision Tree
# ================
dt_params = {
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(class_weight="balanced"), dt_params, cv=5)
dt_grid.fit(X_train, y_train)
dt_pred = dt_grid.predict(X_test)

# ================
# Model: KNN
# ================
knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_train, y_train)
knn_pred = knn_grid.predict(X_test)

# ================
# Model: Random Forest
# ================
rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None]
}
rf_grid = GridSearchCV(RandomForestClassifier(class_weight="balanced"), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
rf_pred = rf_grid.predict(X_test)

# ================
# Model: Gradient Boosting
# ================
gbc_params = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5]
}
gbc_grid = GridSearchCV(GradientBoostingClassifier(), gbc_params, cv=5)
gbc_grid.fit(X_train, y_train)
gbc_pred = gbc_grid.predict(X_test)

# ================
# Ensemble Voting Classifier
# ================
voting_clf = VotingClassifier(
    estimators=[
        ("svm", svm_grid.best_estimator_),
        ("dt", dt_grid.best_estimator_),
        ("knn", knn_grid.best_estimator_),
        ("rf", rf_grid.best_estimator_),
        ("gbc", gbc_grid.best_estimator_)
    ],
    voting="hard"
)
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)

# ================
# Evaluation Function
# ================
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

# ================
# Results
# ================
evaluate_model("SVM", y_test, svm_pred)
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("KNN", y_test, knn_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Gradient Boosting", y_test, gbc_pred)
evaluate_model("Voting Classifier", y_test, voting_pred)




# =================
# for representation
# =================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Helper function to compute metrics
def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]

# Compute metrics for each model
svm_metrics = get_metrics(y_test, svm_pred)
dt_metrics = get_metrics(y_test, dt_pred)
knn_metrics = get_metrics(y_test, knn_pred)

# Metric names
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Convert to numpy array for easier indexing
metric_values = np.array([
    svm_metrics,
    dt_metrics,
    knn_metrics
])

# Bar positions
x = np.arange(len(metrics_names))
bar_width = 0.2

# Create figure
plt.figure(figsize=(10,6))

# Plot bars for each model
plt.bar(x - bar_width, metric_values[0], width=bar_width, label='SVM', color='skyblue')
plt.bar(x, metric_values[1], width=bar_width, label='Decision Tree', color='salmon')
plt.bar(x + bar_width, metric_values[2], width=bar_width, label='KNN', color='lightgreen')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Comparison of Model Performance Metrics')
plt.xticks(x, metrics_names)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()