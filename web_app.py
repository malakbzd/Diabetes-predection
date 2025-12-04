from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diabetes import DiabetesPredictor  # <-- your existing class

app = FastAPI(title="Diabetes Prediction API", version="1.0")

# Allow frontend / JS clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Load Model on Startup
# =========================================================
predictor = DiabetesPredictor()
if not predictor.load_model():
    print("⚠ No saved model found — training a new one...")
    df = predictor.load_and_preprocess_data()
    predictor.train_model(df)
    predictor.save_model()
print("✅ Model loaded and ready.")


# =========================================================
# HTML FORM UI
# =========================================================
@app.get("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Diabetes Predictor</title>

        <style>
            body {
                background: #f2f7ff;
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                padding-top: 40px;
            }

            .container {
                width: 440px;
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }

            h2 {
                text-align: center;
                color: #2867b2;
                margin-bottom: 10px;
            }

            p.description {
                text-align: center;
                font-size: 14px;
                color: #555;
                margin-bottom: 25px;
            }

            label {
                display: block;
                margin-top: 12px;
                font-weight: bold;
                color: #333;
            }

            input, select {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                font-size: 15px;
                border-radius: 6px;
                border: 1px solid #ccc;
                box-sizing: border-box;
            }

            .help-text {
                font-size: 12px;
                color: #666;
                margin-top: 4px;
            }

            button {
                margin-top: 20px;
                width: 100%;
                padding: 12px;
                background: #2867b2;
                border: none;
                color: white;
                font-size: 16px;
                border-radius: 6px;
                cursor: pointer;
                transition: 0.3s;
            }

            button:hover {
                background: #1d4f88;
            }

            footer {
                text-align: center;
                margin-top: 18px;
                color: #777;
                font-size: 13px;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h2>🩺 Diabetes Predictor</h2>
            <p class="description">Enter your health information to estimate diabetes risk.</p>

            <form action="/predict_form" method="post">
                
                <!-- Gender -->
                <label>Gender</label>
                <select name="gender">
                    <option>Female</option>
                    <option>Male</option>
                </select>
                <div class="help-text">
                    Select your biological sex .
                </div>

                <!-- Age -->
                <label>Age</label>
                <input type="number" name="age" step="0.1" required>
                <div class="help-text">
                    Enter your current age .
                </div>

                <!-- BMI -->
                <label>BMI</label>
                <input type="number" name="bmi" step="0.1" required>
                <div class="help-text">
                    BMI (Body Mass Index) = weight(kg) ÷ height(m²).  
                    Example: 70 kg / (1.75 m × 1.75 m) = 22.9  
                </div>

                <!-- Smoking History -->
                <label>Smoking History</label>
                <select name="smoking_history">
                    <option>never</option>
                    <option>not current</option>
                    <option>current</option>
                    <option>No Info</option>
                    <option>ever</option>
                    <option>former</option>
                </select>
                <div class="help-text">
                    Choose the option that best describes your smoking habits.
                </div>

                <!-- HbA1c -->
                <label>HbA1c Level</label>
                <input type="number" name="hbA1c_level" step="0.1" required>
                <div class="help-text">
                    A blood test showing average sugar over 3 months.  
                    Normal: 4.0–5.6% • Prediabetes: 5.7–6.4% • Diabetes: 6.5%+  
                    Enter a value between 3–15.
                </div>

                <!-- Blood Glucose -->
                <label>Blood Glucose Level</label>
                <input type="number" name="blood_glucose_level" step="0.01" required>
                <div class="help-text">
                    Fasting blood sugar measured with a glucometer or lab test.
                </div>

                <label>Unit</label>
                <select name="glucose_unit">
                    <option value="mg/dL">mg/dL (standard)</option>
                    <option value="g/L">g/L (Algeria)</option>
                </select>
                <div class="help-text">
                    Choose the unit of your blood glucose measurement. g/L will be converted automatically to mg/dL.
                </div>


                <button type="submit">Predict Diabetes Risk</button>
            </form>

            <footer>Diabetes Predictor v1.0</footer>
        </div>
    </body>
    </html>
    """




# =========================================================
# POST: API endpoint for form submission
# =========================================================
@app.post("/predict_form", response_class=HTMLResponse)
@app.post("/predict_form", response_class=HTMLResponse)



def predict_form(
    gender: str = Form(...),
    age: float = Form(...),
    bmi: float = Form(...),
    smoking_history: str = Form(...),
    hbA1c_level: float = Form(...),
    blood_glucose_level: float = Form(...),
    glucose_unit: str = Form("mg/dL")
):
    # Convert blood glucose to mg/dL if needed
    def convert_glucose(value: float, unit: str) -> float:
        if unit == "g/L":
            return value * 100  # 1 g/L = 100 mg/dL
        return value

    blood_glucose_mgdl = convert_glucose(blood_glucose_level, glucose_unit)

    user_input = {
        "gender": gender,
        "age": age,
        "bmi": bmi,
        "smoking_history": smoking_history,
        "hbA1c_level": hbA1c_level,
        "blood_glucose_level": blood_glucose_mgdl
    }

    result = predictor.predict_diabetes(user_input)

    # Colors by risk category
    risk_colors = {
        "High": "#e63946",
        "Medium": "#ffb703",
        "Low": "#2a9d8f"
    }
    color = risk_colors.get(result.get("risk_category"), "#333")

    # --- Add advice based on risk ---
    advice_dict = {
        "High": [
            "Consult a doctor immediately for personalized treatment.",
            "Follow a balanced diet low in sugar and refined carbs.",
            "Engage in regular physical activity.",
            "Monitor blood glucose levels daily."
        ],
        "Medium": [
            "Maintain a healthy diet and monitor carbohydrate intake.",
            "Exercise at least 30 minutes daily.",
            "Check blood sugar regularly and schedule regular checkups.",
            "Avoid smoking and limit alcohol consumption."
        ],
        "Low": [
            "Continue a balanced diet and regular exercise.",
            "Monitor your weight and BMI.",
            "Maintain healthy lifestyle habits to prevent diabetes.",
            "Regular check-ups are recommended."
        ]
    }
    patient_advice = advice_dict.get(result.get("risk_category"), ["Follow general healthy lifestyle habits."])

    # Render HTML
    html_result = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                background: #f2f7ff;
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                padding-top: 50px;
            }}
            .result-box {{
                width: 420px;
                background: white;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                border-left: 10px solid {color};
            }}
            h2 {{
                color: #2867b2;
                margin-bottom: 20px;
            }}
            p {{
                font-size: 16px;
                color: #333;
                margin: 8px 0;
            }}
            ul {{
                margin-top: 10px;
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 6px;
                font-size: 14px;
            }}
            a {{
                display: block;
                margin-top: 20px;
                text-align: center;
                background: #2867b2;
                color: white;
                padding: 10px;
                border-radius: 6px;
                text-decoration: none;
                transition: 0.3s;
            }}
            a:hover {{
                background: #1d4f88;
            }}
        </style>
    </head>
    <body>
        <div class="result-box">
            <h2>Prediction Result</h2>
    """

    if "error" in result:
        html_result += f"<p style='color:#e63946; font-weight:bold;'>Error: {result['error']}</p>"
    else:
        html_result += f"""
            <p><b>Prediction:</b> {result['prediction']}</p>
            <p><b>Probability:</b> {result['probability']}%</p>
            <p><b>Risk Category:</b> <span style="color:{color}; font-weight:bold;">{result['risk_category']}</span></p>
            <p><b>Confidence:</b> {result['confidence']}</p>
            <h3>💡 Health Advice:</h3>
            <ul>
        """
        for tip in patient_advice:
            html_result += f"<li>{tip}</li>"
        html_result += "</ul>"

    html_result += """
            <a href="/">⬅ Back to Prediction Form</a>
        </div>
    </body>
    </html>
    """

    return html_result


# =========================================================
# JSON API Endpoint (for external apps)
# =========================================================
@app.post("/predict_json")
def predict_json(data: dict):
    result = predictor.predict_diabetes(data)
    return JSONResponse(result)
