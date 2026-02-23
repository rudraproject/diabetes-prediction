from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = "9f3a8d72c1b5e6f4a9d2b3c7e8f1a0b2"

# ---------------- DATABASE CONFIG ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = (
    "mssql+pyodbc://RUDRA\\prern@RUDRA\\SQLEXPRESS/sugardb?"
    "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- DATABASE MODELS ----------------
class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    Pregnancies = db.Column(db.Integer, nullable=False)
    Glucose = db.Column(db.Integer, nullable=False)
    BloodPressure = db.Column(db.Integer, nullable=False)
    SkinThickness = db.Column(db.Integer, nullable=False)
    Insulin = db.Column(db.Integer, nullable=False)
    BMI = db.Column(db.Float, nullable=False)
    DPF = db.Column(db.Float, nullable=False)
    Age = db.Column(db.Integer, nullable=False)
    result = db.Column(db.String(80), nullable=False)
    confidence = db.Column(db.Float, nullable=False)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


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

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
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


# ✅ Improved Logout
@app.route("/logout")
def logout():
    session.clear()   # Clears entire session securely
    return redirect(url_for("login"))


@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    patients = Patient.query.order_by(Patient.id.desc()).all()
    return render_template("history.html", data=patients)


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
        DPF = float(request.form['DPF'])
        Age = int(request.form['Age'])

        input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                                SkinThickness, Insulin, BMI, DPF, Age]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        confidence = round(probability * 100, 2)

        if prediction == 1:
            result_text = "High Risk of Diabetes"
            message = "Clinical consultation recommended."
            color = "danger"
        else:
            result_text = "Low Risk of Diabetes"
            message = "Lower probability indicators."
            color = "success"

        new_patient = Patient(
            Pregnancies=Pregnancies,
            Glucose=Glucose,
            BloodPressure=BloodPressure,
            SkinThickness=SkinThickness,
            Insulin=Insulin,
            BMI=BMI,
            DPF=DPF,
            Age=Age,
            result=result_text,
            confidence=confidence
        )

        db.session.add(new_patient)
        db.session.commit()

        return render_template(
            "result.html",
            result=result_text,
            message=message,
            confidence=confidence,
            color=color
        )

    except Exception as e:
        return f"Error occurred: {e}"


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        print("Tables in database:", inspector.get_table_names())

    app.run(debug=True, use_reloader=False)
