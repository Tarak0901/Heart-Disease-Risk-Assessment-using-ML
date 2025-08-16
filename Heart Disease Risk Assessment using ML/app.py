from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
import joblib
import numpy as np
import pymongo

# Connect to MongoDB Atlas
client = pymongo.MongoClient("mongodb+srv://eitish:eitish2025@heartcluster.9muxfsf.mongodb.net/?retryWrites=true&w=majority&appName=HeartCluster")
db = client["heart_disease_db"]

# Collections
users_collection = db["users"]
patient_collection = db["patient_data"]

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm_model.pkl")

healthy_ranges = {
    "Age": (30, 60, "Optimal adult age range."),
    "Sex (1=Male, 0=Female)": (0, 1, "Sex is informational, not risk factor itself."),
    "Chest Pain Type": (0, 1, "0=typical angina; lower values are better."),
    "Resting Blood Pressure": (90, 120, "Normal resting BP: 90–120 mmHg."),
    "Cholesterol (mg/dl)": (125, 200, "Desirable cholesterol <200 mg/dl."),
    "Fasting Blood Sugar >120?": (0, 0, "0=Normal fasting sugar."),
    "Resting ECG": (0, 0, "0=Normal ECG."),
    "Max Heart Rate Achieved": (140, 190, "Normal varies by age; higher fitness = higher max HR."),
    "Exercise Induced Angina?": (0, 0, "0=No angina induced by exercise."),
    "Oldpeak": (0, 1, "ST depression <1 is normal."),
    "Slope of ST Segment": (1, 2, "1–2 is normal slope."),
    "Number of Major Vessels": (0, 0, "0 vessels colored is normal."),
    "Thalassemia": (1, 1, "1=Normal thalassemia."),
}

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        role = request.form["role"]
        if role == "doctor":
            return "Doctor registration is not allowed."

        name = request.form["name"]
        mobile = request.form["mobile"]
        email = request.form["email"]
        password = request.form["password"]

        user_data = {
            "name": name,
            "mobile": mobile,
            "email": email,
            "password": password,
            "role": role
        }

        users_collection.insert_one(user_data)
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route('/login_doctor', methods=['POST'])
def login_doctor():
    doctor_id = request.form['doctor_id']
    password = request.form['password']

    # Only one hardcoded doctor login
    if doctor_id == "doctor@example.com" and password == "doctor123":
        session['user'] = doctor_id
        session['role'] = 'doctor'
        return redirect('/doctor_dashboard')
    else:
        return render_template("login.html", error="Invalid doctor credentials!")


@app.route('/login_patient', methods=['POST'])
def login_patient():
    patient_id = request.form['patient_id']
    password = request.form['password']

    patient = users_collection.find_one({
        "email": patient_id,
        "password": password,
        "role": "patient"
    })

    if patient:
        session['user'] = patient['email']
        session['role'] = 'patient'
        return redirect(url_for('patient_dashboard'))
    else:
        return render_template("login.html", error="Invalid patient credentials!")

@app.route("/patient_form")
def patient_form():
    if session.get("role") != "patient":
        return redirect("/")
    return render_template("patient_form.html")

@app.route("/submit_data", methods=["POST"])
def submit_data():
    if session.get("role") != "patient":
        return redirect("/")

    patient_id = session["user"]
    features = [float(request.form[f"feature{i}"]) for i in range(1, 14)]

    submission = {
        "patient_id": patient_id,
        "features": features,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prediction": None,
        "approved": False
    }

    patient_collection.insert_one(submission)

    return redirect(url_for("patient_dashboard"))

@app.route('/doctor_dashboard')
def doctor_dashboard():
    if session.get('role') == 'doctor':
        doctor_email = session['user']
        records = list(patient_collection.find())
        return render_template('doctor_dashboard.html', doctor_email=doctor_email, records=records)
    return redirect('/')

@app.route('/patient_dashboard')
def patient_dashboard():
    if session.get('role') != 'patient':
        return redirect('/')

    patient_id = session['user']
    records = list(patient_collection.find({'patient_id': patient_id}))

    if records:
        FEATURE_NAMES = list(healthy_ranges.keys())
        return render_template("patient_history.html",
                               records=records,
                               patient_id=patient_id,
                               submissions=records,
                               feature_names=FEATURE_NAMES,
                               healthy_ranges=healthy_ranges,
                               zip=zip)
    else:
        return redirect("/patient_form")

from bson import ObjectId

@app.route("/predict/<patient_id>")
def predict(patient_id):
    record = patient_collection.find_one({"patient_id": patient_id}, sort=[("date", -1)])

    if not record:
        return "<h3>No data found for this patient.</h3><a href='/doctor_dashboard'>Back</a>"

    if record.get("prediction") is None:
        features = record["features"]
        input_scaled = scaler.transform([features])
        prediction_value = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        result_text = (
            f"⚠️ High risk of heart disease (probability {probability:.2f})"
            if prediction_value == 1
            else f"✅ Low risk of heart disease (probability {probability:.2f})"
        )

        # Update DB
        patient_collection.update_one(
            {"_id": record["_id"]},
            {"$set": {
                "prediction": result_text,
                "approved": True
            }}
        )
    else:
        result_text = record["prediction"]

    # 🔥 PASS individual fields, including date
    return render_template("result.html",
                           patient_id=patient_id,
                           result=result_text,
                           date=record.get("date", "N/A"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)