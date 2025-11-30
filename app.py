# app.py - Main Flask application
from flask import Flask, render_template, request, jsonify
from predictor import DiabetesPredictor
import json

app = Flask(__name__)

# Initialize predictor
predictor = DiabetesPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        user_data = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'bmi': float(request.form['bmi']),
            'smoking_history': request.form['smoking_history'],
            'hbA1c_level': float(request.form['hba1c']),
            'blood_glucose_level': float(request.form['glucose'])
        }
        
        # Make prediction
        result = predictor.predict_diabetes(user_data)
        
        if 'error' in result:
            return render_template('index.html', error=result['error'])
        
        # Generate health advice based on results
        health_advice = generate_health_advice(user_data, result)
        
        return render_template('results.html', 
                             result=result, 
                             user_data=user_data,
                             advice=health_advice)
                             
    except Exception as e:
        return render_template('index.html', error=str(e))

def generate_health_advice(user_data, result):
    """Generate personalized health advice"""
    advice = []
    
    # BMI-based advice
    bmi = user_data['bmi']
    if bmi > 30:
        advice.append("Consider gradual weight loss through balanced nutrition")
    elif bmi > 25:
        advice.append("Maintain your weight with regular physical activity")
    else:
        advice.append("Great job maintaining a healthy weight!")
    
    # Age-based advice
    age = user_data['age']
    if age > 50:
        advice.append("Regular health check-ups are important at your age")
    elif age > 30:
        advice.append("Now is the perfect time to build healthy lifelong habits")
    else:
        advice.append("Building healthy habits early sets you up for life!")
    
    # Risk-based advice
    risk = result['risk_category']
    if risk == "High":
        advice.append("Consider consulting a healthcare provider for personalized guidance")
        advice.append("Focus on consistent lifestyle changes rather than quick fixes")
    elif risk == "Medium":
        advice.append("Small, consistent improvements can significantly reduce your risk")
        advice.append("Monitor your progress and celebrate small victories")
    else:
        advice.append("Keep up your healthy lifestyle - you're doing great!")
        advice.append("Share your healthy habits with friends and family")
    
    return advice

if __name__ == '__main__':
    # Load model when starting app
    if not predictor.load_model():
        print("❌ Model not found. Please train the model first.")
        print("Run: python train_model.py")
    else:
        print("✅ Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)