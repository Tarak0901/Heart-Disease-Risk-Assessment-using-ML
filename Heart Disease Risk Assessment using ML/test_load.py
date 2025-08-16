import joblib
import numpy as np

# Load
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm_model.pkl")

# Fake input
example_input = np.array([[60,1,3,140,250,0,1,150,0,1.5,2,0,2]])
scaled_input = scaler.transform(example_input)

# Predict
prediction = model.predict(scaled_input)
probability = model.predict_proba(scaled_input)

print("Prediction:", prediction)
print("Probability:", probability)