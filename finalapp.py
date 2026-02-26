from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)

# ---------------- SECRET KEY ----------------
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")

# ---------------- DATABASE CONFIG ----------------
database_url = os.environ.get("DATABASE_URL")

if not database_url:
    raise ValueError("DATABASE_URL is not set in environment variables")

# Fix Render postgres:// issue
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- DATABASE MODELS ----------------

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)

    Pregnancies = db.Column(db.Integer, nullable=False)
    Glucose = db.Column(db.Integer, nullable=False)
    BloodPressure = db.Column(db.Integer, nullable=False)
    SkinThickness = db.Column(db.Integer, nullable=False)
    Insulin = db.Column(db.Integer, nullable=False)
    BMI = db.Column(db.Float, nullable=False)
    Age = db.Column(db.Integer, nullable=False)

    risk_level = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timeline = db.Column(db.String(200), nullable=False)


# ---------------- CREATE TABLES ----------------
with app.app_context():
    db.create_all()

# ---------------- LOAD MODEL ----------------
model = joblib.load("diabetes_m.pkl")
scaler = joblib.load("diabetes_sc.pkl")


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            return "Username already exists!"

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user"] = user.username
            return redirect(url_for("home"))
        else:
            return "Invalid Username or Password"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    patients = Patient.query.filter_by(username=session["user"]) \
        .order_by(Patient.id.desc()).all()

    return render_template("history.html", data=patients)


# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        Age = int(request.form['Age'])

        input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                                SkinThickness, Insulin, BMI, Age]])

        input_scaled = scaler.transform(input_data)

        probability = float(model.predict_proba(input_scaled)[0][1])
        confidence = round(probability * 100, 2)

        recommendations = []

        # ---------------- RISK LOGIC ----------------

        if probability < 0.25:
            risk_level = "Low Risk"
            color = "success"
            timeline = "Low short-term risk (0–3 years). Maintain healthy lifestyle."

            recommendations.extend([
                "Maintain balanced low-glycemic diet.",
                "Exercise 30 minutes daily.",
                "Annual fasting glucose test.",
                "Maintain healthy BMI range.",
                "Ensure proper sleep cycle."
            ])

        elif 0.25 <= probability < 0.50:
            risk_level = "Pre-Diabetic Risk"
            color = "warning"
            timeline = "Moderate progression risk within 3–7 years without intervention."

            recommendations.extend([
                "Adopt structured low-carb diet.",
                "Increase physical activity to 45 minutes daily.",
                "Monitor blood glucose every 3–6 months.",
                "Reduce weight by 5–10%.",
                "Consult doctor for HbA1c evaluation."
            ])

        else:
            risk_level = "High Risk"
            color = "danger"
            timeline = "High likelihood of progression in near future without medical care."

            recommendations.extend([
                "Consult endocrinologist immediately.",
                "Perform HbA1c and fasting tests.",
                "Begin medically supervised diet plan.",
                "Daily glucose monitoring advised.",
                "Structured weight reduction program."
            ])

        # Extra smart checks
        if Glucose > 140:
            recommendations.append("Elevated glucose detected: reduce refined sugars.")

        if BMI >= 30:
            recommendations.append("High BMI detected: initiate weight-loss program.")

        if BloodPressure > 90:
            recommendations.append("Elevated blood pressure: adopt low-sodium diet.")

        if Insulin > 180:
            recommendations.append("Possible insulin resistance detected.")

        if Age > 45:
            recommendations.append("Annual metabolic screening recommended.")

        # Save record
        new_patient = Patient(
            username=session["user"],
            Pregnancies=Pregnancies,
            Glucose=Glucose,
            BloodPressure=BloodPressure,
            SkinThickness=SkinThickness,
            Insulin=Insulin,
            BMI=BMI,
            Age=Age,
            risk_level=risk_level,
            confidence=confidence,
            timeline=timeline
        )

        db.session.add(new_patient)
        db.session.commit()

        return render_template(
            "result.html",
            risk=risk_level,
            confidence=confidence,
            timeline=timeline,
            color=color,
            recommendations=recommendations
        )

    except Exception as e:
        db.session.rollback()
        return f"Error occurred: {str(e)}"


# ---------------- MAIN ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
