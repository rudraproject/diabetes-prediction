from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)

# ---------------- CONFIGURATION ----------------
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "devkey123")

# ---------------- DATABASE CONFIG ----------------
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///local.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# FIX FOR SSL/EOF ERRORS:
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_pre_ping": True,  # Checks if connection is alive before using it
    "pool_recycle": 300,    # Restart connections every 5 minutes
    "pool_size": 10,        # Max connections
    "max_overflow": 20,     # Extra connections if busy
}

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

# Create tables within the app context
with app.app_context():
    db.create_all()

# ---------------- LOAD ML MODELS ----------------
try:
    model = joblib.load("diabetes_m.pkl")
    scaler = joblib.load("diabetes_sc.pkl")
except Exception as e:
    print(f"Warning: Model files not found or failed to load: {e}")
    model = None
    scaler = None

# ---------------- ROUTES ----------------

@app.route("/health")
def health_check():
    """Route for Render to verify the app is listening."""
    return {"status": "healthy"}, 200

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
        try:
            if User.query.filter_by(username=username).first():
                return "Username already exists!"
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            return f"Registration Error: {str(e)}"
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        try:
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                session["user"] = user.username
                return redirect(url_for("home"))
            return "Invalid Username or Password"
        except Exception as e:
            return f"Login Error: {str(e)}"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))
    patients = Patient.query.filter_by(username=session["user"]).order_by(Patient.id.desc()).all()
    return render_template("history.html", data=patients)

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    if model is None or scaler is None:
        return "Model not loaded. Please check server logs."

    try:
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        Age = int(request.form['Age'])

        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age]])
        input_scaled = scaler.transform(input_data)
        probability = float(model.predict_proba(input_scaled)[0][1])
        confidence = round(probability * 100, 2)

        recommendations = []
        if probability < 0.25:
            risk_level, color, timeline = "Low Risk", "success", "Low short-term risk (0–3 years). Maintain healthy lifestyle."
            recommendations.extend(["Maintain balanced low-glycemic diet.", "Exercise 30 minutes daily.", "Annual fasting glucose test."])
        elif 0.25 <= probability < 0.50:
            risk_level, color, timeline = "Pre-Diabetic Risk", "warning", "Moderate progression risk within 3–7 years without intervention."
            recommendations.extend(["Adopt structured low-carb diet.", "Monitor blood glucose every 3–6 months."])
        else:
            risk_level, color, timeline = "High Risk", "danger", "High likelihood of progression in near future without medical care."
            recommendations.extend(["Consult endocrinologist immediately.", "Daily glucose monitoring advised."])

        # Extra smart checks
        if Glucose > 140: recommendations.append("Elevated glucose detected: reduce refined sugars.")
        if BMI >= 30: recommendations.append("High BMI detected: initiate weight-loss program.")

        new_patient = Patient(
            username=session["user"], Pregnancies=Pregnancies, Glucose=Glucose,
            BloodPressure=BloodPressure, SkinThickness=SkinThickness, Insulin=Insulin,
            BMI=BMI, Age=Age, risk_level=risk_level, confidence=confidence, timeline=timeline
        )
        db.session.add(new_patient)
        db.session.commit()

        return render_template("result.html", risk=risk_level, confidence=confidence, timeline=timeline, color=color, recommendations=recommendations)

    except Exception as e:
        db.session.rollback()
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)