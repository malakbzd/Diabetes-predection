from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from datetime import datetime
import os
from diabetes import DiabetesPredictor

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Risk Assessment",
    version="1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize predictor
predictor = DiabetesPredictor()

# Load or train model
if not predictor.load_model():
    print("⚠ No saved model found — training a new one...")
    try:
        df = predictor.load_and_preprocess_data("diabetes_dataset.csv")
        predictor.train_model(df)
        predictor.save_model()
        print("✅ Model trained and saved successfully!")
    except Exception as e:
        print(f"❌ Failed to train model: {e}")

# Helper functions
def convert_glucose(value, unit):
    """Convert glucose units"""
    try:
        value = float(value)
        if unit == "mmol":
            return value * 18  # Convert mmol/L to mg/dL
        return value
    except:
        return value

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight", "blue", "BMI below 18.5"
    elif bmi < 25:
        return "Normal", "green", "Healthy weight (18.5-24.9)"
    elif bmi < 30:
        return "Overweight", "orange", "BMI 25-29.9"
    else:
        return "Obese", "red", "BMI 30 or higher"

def get_risk_color(probability):
    """Get color based on risk probability"""
    if probability < 0.3:
        return "success"
    elif probability < 0.6:
        return "warning"
    else:
        return "danger"

def generate_risk_meter(probability):
    """Generate risk meter HTML"""
    width = probability * 100
    color = "#28a745" if probability < 0.3 else "#ffc107" if probability < 0.6 else "#dc3545"
    
    return f"""
    <div class="progress" style="height: 30px;">
        <div class="progress-bar" role="progressbar" style="width: {width}%; background-color: {color};" 
             aria-valuenow="{width}" aria-valuemin="0" aria-valuemax="100">
            {probability:.1%} Risk
        </div>
    </div>
    """

# Create templates for different pages

# 1. HOME PAGE
home_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Assessment - Home</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
background: linear-gradient(#a6d1ff 50%, #4da6ff 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .hero-section {
            background: linear-gradient(rgba(44, 62, 80, 0.9), rgba(44, 62, 80, 0.8)), 
                        url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');
            background-size: cover;
            background-position: center;
            color: white;
            padding: 100px 0;
            margin-top: 70px;
        }
        
        .navbar {
            background: rgba(44, 62, 80, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: none;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3498db, #2980b9);
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .feature-card {
            text-align: center;
            padding: 30px 20px;
            border-radius: 15px;
            color: white;
            margin: 10px;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .feature-card.blue { background: linear-gradient(135deg, #3498db, #2980b9); }
        .feature-card.green { background: linear-gradient(135deg, #2ecc71, #27ae60); }
        .feature-card.purple { background: linear-gradient(135deg, #9b59b6, #8e44ad); }
        .feature-card.orange { background: linear-gradient(135deg, #e67e22, #d35400); }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-heartbeat me-2"></i>
                <strong>DiaRisk AI</strong>
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link active" href="/"><i class="fas fa-home me-1"></i> Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/calculator"><i class="fas fa-calculator me-1"></i> Risk Calculator</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/articles"><i class="fas fa-book me-1"></i> Diabetes Articles</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <div class="hero-section">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-8">
                    <h1 class="display-4 fw-bold mb-4">Take Control of Your Health</h1>
                    <p class="lead mb-4">
                        Early detection of diabetes can prevent serious complications. 
                        Our AI-powered tool assesses your risk in minutes and provides personalized recommendations.
                    </p>
                    <div class="d-flex gap-3">
                        <a href="/calculator" class="btn btn-light btn-lg">
                            <i class="fas fa-play-circle me-2"></i>Start Assessment
                        </a>
                        <a href="/articles" class="btn btn-outline-light btn-lg">
                            <i class="fas fa-book me-2"></i>Learn About Diabetes
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container mt-5">
        <!-- Why It Matters -->
        <div class="row mb-5">
            <div class="col-12 text-center mb-4">
                <h2>Why Diabetes Screening Matters</h2>
                <p class="lead">Diabetes affects millions worldwide, but early detection can save lives.</p>
            </div>
            
            <div class="col-md-3">
                <div class="feature-card blue">
                    <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                    <h4>Early Warning</h4>
                    <p>Detect prediabetes before it becomes full diabetes</p>
                </div>
            </div>
            
            <div class="col-md-3">
                <div class="feature-card green">
                    <i class="fas fa-shield-alt fa-3x mb-3"></i>
                    <h4>Prevention</h4>
                    <p>70% of Type 2 diabetes cases are preventable</p>
                </div>
            </div>
            
            <div class="col-md-3">
                <div class="feature-card purple">
                    <i class="fas fa-brain fa-3x mb-3"></i>
                    <h4>AI-Powered</h4>
                    <p>Machine learning algorithms for accurate assessment</p>
                </div>
            </div>
            
            <div class="col-md-3">
                <div class="feature-card orange">
                    <i class="fas fa-user-md fa-3x mb-3"></i>
                    <h4>Expert Guidance</h4>
                    <p>Personalized recommendations based on your profile</p>
                </div>
            </div>
        </div>

        <!-- Global Statistics -->
        <div class="card mb-5">
            <div class="card-header bg-dark text-white">
                <h3 class="mb-0"><i class="fas fa-globe me-2"></i>Diabetes Worldwide</h3>
            </div>
            <div class="card-body">
                <div class="row text-center">
                    <div class="col-md-3">
                        <div class="stat-number text-primary">463M</div>
                        <p>People with diabetes</p>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-number text-warning">232M</div>
                        <p>Undiagnosed cases</p>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-number text-danger">1.5M</div>
                        <p>Deaths annually</p>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-number text-success">70%</div>
                        <p>Preventable cases</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- How It Works -->
        <div class="row mb-5">
            <div class="col-12 text-center mb-4">
                <h2>How It Works</h2>
            </div>
            
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <div class="mb-3">
                            <span class="badge bg-primary rounded-circle p-3" style="font-size: 1.5rem;">1</span>
                        </div>
                        <h4>Enter Your Info</h4>
                        <p>Provide basic health information like age, BMI, and blood sugar levels</p>
                        <i class="fas fa-edit fa-2x text-primary mt-3"></i>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <div class="mb-3">
                            <span class="badge bg-primary rounded-circle p-3" style="font-size: 1.5rem;">2</span>
                        </div>
                        <h4>AI Analysis</h4>
                        <p>Our machine learning model analyzes your risk factors</p>
                        <i class="fas fa-brain fa-2x text-primary mt-3"></i>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <div class="mb-3">
                            <span class="badge bg-primary rounded-circle p-3" style="font-size: 1.5rem;">3</span>
                        </div>
                        <h4>Get Results</h4>
                        <p>Receive personalized risk assessment and recommendations</p>
                        <i class="fas fa-chart-line fa-2x text-primary mt-3"></i>
                    </div>
                </div>
            </div>
        </div>

        <!-- Call to Action -->
        <div class="card bg-dark text-white mb-5">
            <div class="card-body text-center p-5">
                <h2 class="mb-4">Ready to Check Your Risk?</h2>
                <p class="lead mb-4">It takes less than 2 minutes to get your personalized diabetes risk assessment.</p>
                <a href="/calculator" class="btn btn-light btn-lg">
                    <i class="fas fa-play-circle me-2"></i>Start Free Assessment Now
                </a>
                <p class="mt-3 small opacity-75">No registration required • Completely confidential</p>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center text-white py-4">
            <p class="mb-0">© 2024 DiaRisk AI - Diabetes Risk Assessment Tool</p>
            <p class="small opacity-75">For educational purposes only. Always consult healthcare professionals for medical advice.</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Smooth scroll
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
    </script>
</body>
</html>"""

# 2. CALCULATOR PAGE
calculator_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Calculator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(#a6d1ff 50%, #4da6ff 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .navbar {
            background: rgba(44, 62, 80, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: none;
            margin-bottom: 20px;
        }
        
        .card-header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            border-radius: 15px 15px 0 0 !important;
            padding: 20px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3498db, #2980b9);
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .form-control, .form-select {
            border-radius: 10px;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
        }
        
        .form-control:focus, .form-select:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
        }
        
        .info-icon {
            color: #3498db;
            cursor: help;
            margin-left: 5px;
        }
        
        .info-popup {
            display: none;
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            z-index: 1000;
            max-width: 300px;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-heartbeat me-2"></i>
                <strong>DiaRisk AI</strong>
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/"><i class="fas fa-home me-1"></i> Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/calculator"><i class="fas fa-calculator me-1"></i> Risk Calculator</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/articles"><i class="fas fa-book me-1"></i> Diabetes Articles</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-5" style="padding-top: 80px;">
        <!-- Calculator Form -->
        <div class="card mb-5">
            <div class="card-header">
                <h3 class="mb-0"><i class="fas fa-calculator me-2"></i>Diabetes Risk Calculator</h3>
                <p class="mb-0 opacity-75">Enter your health information for personalized risk assessment</p>
            </div>
            <div class="card-body">
                <form action="/predict" method="post" id="predictionForm">
                    <div class="row">
                        <!-- Personal Information -->
                        <div class="col-md-6">
                            <h5 class="border-bottom pb-2 mb-4"><i class="fas fa-user me-2"></i>Personal Information</h5>
                            
                            <div class="mb-3">
                                <label class="form-label">
                                    Gender <span class="text-danger">*</span>
                                    <i class="fas fa-info-circle info-icon" 
                                       data-bs-toggle="tooltip" 
                                       title="Your biological sex affects diabetes risk"></i>
                                </label>
                                <select class="form-select" name="gender" required>
                                    <option value="">Select Gender</option>
                                    <option value="Female">Female</option>
                                    <option value="Male">Male</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">
                                    Age <span class="text-danger">*</span>
                                    <i class="fas fa-info-circle info-icon" 
                                       data-bs-toggle="tooltip" 
                                       title="Diabetes risk increases significantly after age 45"></i>
                                </label>
                                <input type="number" class="form-control" name="age" 
                                       min="0" max="120" step="1" required placeholder="e.g., 45">
                                <div class="form-text">Risk increases after age 45</div>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">
                                    Smoking History <span class="text-danger">*</span>
                                    <i class="fas fa-info-circle info-icon" 
                                       data-bs-toggle="tooltip" 
                                       title="Smoking increases insulin resistance and diabetes risk"></i>
                                </label>
                                <select class="form-select" name="smoking_history" required>
                                    <option value="">Select smoking status</option>
                                    <option value="never">Never smoked</option>
                                    <option value="not current">Not current smoker</option>
                                    <option value="current">Current smoker</option>
                                    <option value="former">Former smoker</option>
                                    <option value="ever">Ever smoked</option>
                                    <option value="no info">No information</option>
                                </select>
                                <div class="form-text">Smoking increases diabetes risk by 30-40%</div>
                            </div>
                        </div>
                        
                        <!-- Health Metrics -->
                        <div class="col-md-6">
                            <h5 class="border-bottom pb-2 mb-4"><i class="fas fa-heartbeat me-2"></i>Health Metrics</h5>
                            
                            <div class="mb-3">
                                <label class="form-label">
                                    BMI (Body Mass Index) <span class="text-danger">*</span>
                                    <i class="fas fa-info-circle info-icon" 
                                       data-bs-toggle="tooltip" 
                                       title="Weight relative to height. Higher BMI = higher diabetes risk"></i>
                                </label>
                                <input type="number" class="form-control" name="bmi" 
                                       min="10" max="60" step="0.1" required placeholder="e.g., 24.5">
                                <div class="form-text">
                                    <span class="badge bg-success">Normal: 18.5-24.9</span>
                                    <span class="badge bg-warning ms-1">Overweight: 25-29.9</span>
                                    <span class="badge bg-danger ms-1">Obese: 30+</span>
                                </div>
                                <button type="button" class="btn btn-sm btn-outline-primary mt-2" 
                                        data-bs-toggle="modal" data-bs-target="#bmiModal">
                                    <i class="fas fa-calculator me-1"></i>Calculate BMI
                                </button>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">
                                    HbA1c Level <span class="text-danger">*</span>
                                    <i class="fas fa-info-circle info-icon" 
                                       data-bs-toggle="tooltip" 
                                       title="Average blood sugar over last 2-3 months. Most important diabetes marker"></i>
                                </label>
                                <input type="number" class="form-control" name="hbA1c_level" 
                                       min="3" max="15" step="0.1" required placeholder="e.g., 5.7">
                                <div class="form-text">
                                    <span class="badge bg-success">Normal: &lt;5.7%</span>
                                    <span class="badge bg-warning ms-1">Prediabetes: 5.7-6.4%</span>
                                    <span class="badge bg-danger ms-1">Diabetes: >=6.5%</span>
                                </div>
                                <button type="button" class="btn btn-sm btn-outline-primary mt-2" 
                                        data-bs-toggle="modal" data-bs-target="#hba1cModal">
                                    <i class="fas fa-info-circle me-1"></i>What is HbA1c?
                                </button>
                            </div>
                            
                            <div class="mb-4">
                                <label class="form-label">
                                    Blood Glucose Level <span class="text-danger">*</span>
                                    <i class="fas fa-info-circle info-icon" 
                                       data-bs-toggle="tooltip" 
                                       title="Current blood sugar level. Measured after 8+ hours fasting"></i>
                                </label>
                                <div class="input-group">
                                    <input type="number" class="form-control" name="blood_glucose_level" 
                                           min="50" max="300" step="1" required placeholder="e.g., 100">
                                    <select class="form-select" style="max-width: 150px;" name="glucose_unit">
                                        <option value="mgdl">mg/dL</option>
                                    </select>
                                </div>
                                <div class="form-text">
                                    <span class="badge bg-success">Normal: 70-99 mg/dL</span>
                                    <span class="badge bg-warning ms-1">Prediabetes: 100-125</span>
                                    <span class="badge bg-danger ms-1">Diabetes: >=126</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        <strong>Note:</strong> This tool provides risk assessment only. It is not a medical diagnosis. 
                        Always consult with healthcare professionals for accurate diagnosis and treatment.
                    </div>
                    
                    <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                        <button type="reset" class="btn btn-outline-secondary me-2">
                            <i class="fas fa-redo me-1"></i>Reset
                        </button>
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="fas fa-chart-line me-2"></i>Calculate My Risk
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <!-- BMI Calculator Modal -->
        <div class="modal fade" id="bmiModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title"><i class="fas fa-calculator me-2"></i>BMI Calculator</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p><strong>BMI (Body Mass Index)</strong> measures body fat based on height and weight.</p>
                        <div class="mb-3">
                            <label>Weight (kg)</label>
                            <input type="number" class="form-control" id="weightInput" placeholder="e.g., 70">
                        </div>
                        <div class="mb-3">
                            <label>Height (cm)</label>
                            <input type="number" class="form-control" id="heightInput" placeholder="e.g., 170">
                        </div>
                        <button class="btn btn-primary" onclick="calculateBMI()">
                            Calculate BMI
                        </button>
                        <div id="bmiResult" class="mt-3" style="display: none;">
                            <h5>Your BMI: <span id="bmiValue"></span></h5>
                            <p id="bmiCategory" class="mb-0"></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- HbA1c Info Modal -->
        <div class="modal fade" id="hba1cModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title"><i class="fas fa-info-circle me-2"></i>What is HbA1c?</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <h6>Hemoglobin A1c (HbA1c)</h6>
                        <p>HbA1c is a blood test that shows your average blood sugar level over the past 2-3 months.</p>
                        
                        <h6 class="mt-3">Why It's Important:</h6>
                        <ul>
                            <li>Primary diagnostic test for diabetes</li>
                            <li>Shows long-term blood sugar control</li>
                            <li>Not affected by recent meals or daily fluctuations</li>
                            <li>Used to monitor diabetes treatment effectiveness</li>
                        </ul>
                        
                        <h6 class="mt-3">Interpretation:</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>HbA1c Level</th>
                                        <th>Category</th>
                                        <th>Meaning</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr class="table-success">
                                        <td>Below 5.7%</td>
                                        <td>Normal</td>
                                        <td>Healthy blood sugar control</td>
                                    </tr>
                                    <tr class="table-warning">
                                        <td>5.7% to 6.4%</td>
                                        <td>Prediabetes</td>
                                        <td>High risk of developing diabetes</td>
                                    </tr>
                                    <tr class="table-danger">
                                        <td>6.5% or higher</td>
                                        <td>Diabetes</td>
                                        <td>Diagnosis of diabetes</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center text-white mt-5 py-4">
            <p class="mb-0">© 2024 DiaRisk AI - Diabetes Risk Assessment Tool</p>
            <p class="small opacity-75">For educational purposes only. Not a medical diagnosis.</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
        const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        })
        
        // BMI Calculator
        function calculateBMI() {
            const weight = parseFloat(document.getElementById('weightInput').value);
            const height = parseFloat(document.getElementById('heightInput').value);
            
            if (weight && height) {
                const heightM = height / 100;
                const bmi = weight / (heightM * heightM);
                const roundedBMI = bmi.toFixed(1);
                
                document.getElementById('bmiValue').textContent = roundedBMI;
                
                let category = '';
                let color = '';
                if (bmi < 18.5) {
                    category = 'Underweight';
                    color = 'primary';
                } else if (bmi < 25) {
                    category = 'Normal weight';
                    color = 'success';
                } else if (bmi < 30) {
                    category = 'Overweight';
                    color = 'warning';
                } else {
                    category = 'Obese';
                    color = 'danger';
                }
                
                document.getElementById('bmiCategory').innerHTML = 
                    `<span class="badge bg-${color}">${category}</span>`;
                document.getElementById('bmiResult').style.display = 'block';
                
                // Update main form BMI field
                document.querySelector('input[name="bmi"]').value = roundedBMI;
            } else {
                alert('Please enter both weight and height');
            }
        }
        
        // Form validation
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            const inputs = this.querySelectorAll('input[required], select[required]');
            let valid = true;
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    valid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    </script>
</body>
</html>"""

# 3. ARTICLES PAGE
articles_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Information & Articles</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(#a6d1ff 50%, #4da6ff 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .navbar {
            background: rgba(44, 62, 80, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .article-card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: none;
            margin-bottom: 30px;
            transition: transform 0.3s ease;
            overflow: hidden;
        }
        
        .article-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .article-header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            border-radius: 15px 15px 0 0;
        }
        
        .article-body {
            padding: 25px;
            background: white;
        }
        
        .category-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            z-index: 10;
        }
        
        .article-image {
            height: 200px;
            object-fit: cover;
            width: 100%;
        }
        
        .article-meta {
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 15px;
        }
        
        .read-more-btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            border: none;
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .read-more-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .articles-hero {
            background: linear-gradient(rgba(44, 62, 80, 0.9), rgba(44, 62, 80, 0.8));
            color: white;
            padding: 80px 0 40px 0;
            margin-top: 70px;
            border-radius: 0 0 20px 20px;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-heartbeat me-2"></i>
                <strong>DiaRisk AI</strong>
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/"><i class="fas fa-home me-1"></i> Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/calculator"><i class="fas fa-calculator me-1"></i> Risk Calculator</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/articles"><i class="fas fa-book me-1"></i> Diabetes Articles</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <div class="articles-hero">
        <div class="container">
            <div class="row">
                <div class="col-lg-8">
                    <h1 class="display-5 fw-bold mb-3">Diabetes Information Center</h1>
                    <p class="lead mb-4">
                        Comprehensive resources about diabetes prevention, management, and latest research.
                        Stay informed to take control of your health.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <div class="container mt-5">
        <!-- Articles Grid -->
        <div class="row">
            <!-- Article 1 -->
            <div class="col-lg-4 col-md-6">
                <div class="article-card">
                    <span class="badge bg-success category-badge">Basics</span>
                    <div class="article-header">
                        <h4>What is Diabetes?</h4>
                        <p class="mb-0 opacity-75">Understanding the basics</p>
                    </div>
                    <div class="article-body">
                        <div class="article-meta">
                            <i class="far fa-clock me-1"></i> 5 min read • 
                            <i class="far fa-calendar-alt ms-2 me-1"></i> Updated: Dec 2024
                        </div>
                        <p>
                            Diabetes is a chronic disease that occurs when the pancreas cannot produce enough insulin 
                            or when the body cannot effectively use the insulin it produces. Insulin is a hormone 
                            that regulates blood sugar...
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-primary">Type 1 & 2</span>
                            <button class="btn btn-sm read-more-btn" onclick="showArticle(1)">Read More</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Article 2 -->
            <div class="col-lg-4 col-md-6">
                <div class="article-card">
                    <span class="badge bg-warning category-badge">Prevention</span>
                    <div class="article-header">
                        <h4>Preventing Type 2 Diabetes</h4>
                        <p class="mb-0 opacity-75">Lifestyle changes that work</p>
                    </div>
                    <div class="article-body">
                        <div class="article-meta">
                            <i class="far fa-clock me-1"></i> 7 min read • 
                            <i class="far fa-calendar-alt ms-2 me-1"></i> Updated: Nov 2024
                        </div>
                        <p>
                            Up to 70% of type 2 diabetes cases can be prevented or delayed through lifestyle changes. 
                            Regular physical activity, maintaining a healthy weight, and balanced nutrition are key factors...
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-success">Healthy Living</span>
                            <button class="btn btn-sm read-more-btn" onclick="showArticle(2)">Read More</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Article 3 -->
            <div class="col-lg-4 col-md-6">
                <div class="article-card">
                    <span class="badge bg-danger category-badge">Symptoms</span>
                    <div class="article-header">
                        <h4>Early Warning Signs</h4>
                        <p class="mb-0 opacity-75">Recognizing diabetes symptoms</p>
                    </div>
                    <div class="article-body">
                        <div class="article-meta">
                            <i class="far fa-clock me-1"></i> 6 min read • 
                            <i class="far fa-calendar-alt ms-2 me-1"></i> Updated: Oct 2024
                        </div>
                        <p>
                            Early detection of diabetes symptoms can prevent complications. Common signs include 
                            increased thirst, frequent urination, extreme fatigue, blurred vision, and slow-healing 
                            wounds. Many people have prediabetes for years without symptoms...
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-info">Early Detection</span>
                            <button class="btn btn-sm read-more-btn" onclick="showArticle(3)">Read More</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Article 4 -->
            <div class="col-lg-4 col-md-6">
                <div class="article-card">
                    <span class="badge bg-info category-badge">Nutrition</span>
                    <div class="article-header">
                        <h4>Diabetes-Friendly Diet</h4>
                        <p class="mb-0 opacity-75">What to eat and avoid</p>
                    </div>
                    <div class="article-body">
                        <div class="article-meta">
                            <i class="far fa-clock me-1"></i> 8 min read • 
                            <i class="far fa-calendar-alt ms-2 me-1"></i> Updated: Sep 2024
                        </div>
                        <p>
                            A balanced diet is crucial for diabetes management. Focus on whole grains, lean proteins, 
                            healthy fats, and plenty of vegetables. Limit processed foods, sugar-sweetened beverages, 
                            and refined carbohydrates...
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-success">Healthy Eating</span>
                            <button class="btn btn-sm read-more-btn" onclick="showArticle(4)">Read More</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Article 5 -->
            <div class="col-lg-4 col-md-6">
                <div class="article-card">
                    <span class="badge bg-primary category-badge">Treatment</span>
                    <div class="article-header">
                        <h4>Modern Diabetes Treatments</h4>
                        <p class="mb-0 opacity-75">Medications and technologies</p>
                    </div>
                    <div class="article-body">
                        <div class="article-meta">
                            <i class="far fa-clock me-1"></i> 10 min read • 
                            <i class="far fa-calendar-alt ms-2 me-1"></i> Updated: Aug 2024
                        </div>
                        <p>
                            Diabetes treatment has evolved significantly. From oral medications to insulin pumps 
                            and continuous glucose monitors, modern technology helps patients manage their 
                            condition more effectively than ever before...
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-warning">Technology</span>
                            <button class="btn btn-sm read-more-btn" onclick="showArticle(5)">Read More</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Article 6 -->
            <div class="col-lg-4 col-md-6">
                <div class="article-card">
                    <span class="badge bg-secondary category-badge">Research</span>
                    <div class="article-header">
                        <h4>Latest Research & Breakthroughs</h4>
                        <p class="mb-0 opacity-75">Future of diabetes care</p>
                    </div>
                    <div class="article-body">
                        <div class="article-meta">
                            <i class="far fa-clock me-1"></i> 9 min read • 
                            <i class="far fa-calendar-alt ms-2 me-1"></i> Updated: Jul 2024
                        </div>
                        <p>
                            Recent advancements include artificial pancreas systems, stem cell research for 
                            beta cell regeneration, and new drug classes that target different pathways 
                            in glucose metabolism. The future of diabetes care looks promising...
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-danger">Innovation</span>
                            <button class="btn btn-sm read-more-btn" onclick="showArticle(6)">Read More</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Full Article Modal -->
        <div class="modal fade" id="articleModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="articleModalTitle"></h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="articleModalContent">
                        <!-- Article content will be loaded here -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center text-white mt-5 py-4">
            <p class="mb-0">© 2024 DiaRisk AI - Diabetes Education Center</p>
            <p class="small opacity-75">Educational content based on medical guidelines and research</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Articles data
        const articles = {
            1: {
                title: "What is Diabetes? - Complete Guide",
                content: `
                    <h4>Understanding Diabetes</h4>
                    <p>Diabetes mellitus, commonly known as diabetes, is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period.</p>
                    
                    <h5 class="mt-4">Types of Diabetes:</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-header bg-primary text-white">
                                    <h6 class="mb-0">Type 1 Diabetes</h6>
                                </div>
                                <div class="card-body">
                                    <p><strong>Cause:</strong> Autoimmune destruction of insulin-producing beta cells</p>
                                    <p><strong>Onset:</strong> Usually in childhood or adolescence</p>
                                    <p><strong>Treatment:</strong> Insulin injections required</p>
                                    <p><strong>Prevalence:</strong> 5-10% of all diabetes cases</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-header bg-warning text-white">
                                    <h6 class="mb-0">Type 2 Diabetes</h6>
                                </div>
                                <div class="card-body">
                                    <p><strong>Cause:</strong> Insulin resistance and relative insulin deficiency</p>
                                    <p><strong>Onset:</strong> Usually in adulthood (increasingly in children)</p>
                                    <p><strong>Treatment:</strong> Lifestyle changes, oral medications, sometimes insulin</p>
                                    <p><strong>Prevalence:</strong> 90-95% of all diabetes cases</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">Key Statistics:</h5>
                    <ul>
                        <li>463 million adults worldwide have diabetes (2019)</li>
                        <li>Projected to rise to 700 million by 2045</li>
                        <li>1 in 2 adults with diabetes is undiagnosed</li>
                        <li>Diabetes causes 4.2 million deaths annually</li>
                        <li>Healthcare costs: $760 billion globally (2019)</li>
                    </ul>
                    
                    <h5 class="mt-4">The Role of Insulin:</h5>
                    <p>Insulin is a hormone produced by the pancreas that allows cells to absorb glucose from the bloodstream for energy. In diabetes, this process is disrupted, leading to high blood sugar levels.</p>
                `
            },
            2: {
                title: "Preventing Type 2 Diabetes - Evidence-Based Strategies",
                content: `
                    <h4>How to Prevent Type 2 Diabetes</h4>
                    <p>Type 2 diabetes is largely preventable through lifestyle modifications. Research shows that even modest changes can reduce risk by 40-70%.</p>
                    
                    <h5 class="mt-4">1. Maintain Healthy Weight</h5>
                    <p>Losing just 5-10% of body weight can significantly reduce diabetes risk:</p>
                    <ul>
                        <li>BMI between 18.5-24.9 is ideal</li>
                        <li>Waist circumference: < 102 cm (men), < 88 cm (women)</li>
                        <li>Even 5kg weight loss makes a difference</li>
                    </ul>
                    
                    <h5 class="mt-4">2. Regular Physical Activity</h5>
                    <p>Exercise improves insulin sensitivity:</p>
                    <ul>
                        <li>150 minutes moderate exercise weekly (30 mins, 5 days)</li>
                        <li>Include both aerobic and resistance training</li>
                        <li>Break up sitting time every 30 minutes</li>
                        <li>Activities: brisk walking, cycling, swimming, dancing</li>
                    </ul>
                    
                    <h5 class="mt-4">3. Healthy Eating Patterns</h5>
                    <p>Focus on nutrient-dense foods:</p>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card bg-success text-white mb-3">
                                <div class="card-body">
                                    <h6><i class="fas fa-check-circle me-2"></i>Eat More:</h6>
                                    <ul>
                                        <li>Whole grains (brown rice, quinoa, oats)</li>
                                        <li>Non-starchy vegetables</li>
                                        <li>Lean proteins (fish, chicken, legumes)</li>
                                        <li>Healthy fats (avocado, nuts, olive oil)</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card bg-danger text-white mb-3">
                                <div class="card-body">
                                    <h6><i class="fas fa-times-circle me-2"></i>Limit or Avoid:</h6>
                                    <ul>
                                        <li>Sugar-sweetened beverages</li>
                                        <li>Processed meats</li>
                                        <li>Refined carbohydrates</li>
                                        <li>Trans fats</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">4. Other Important Factors</h5>
                    <ul>
                        <li><strong>Sleep:</strong> 7-9 hours quality sleep nightly</li>
                        <li><strong>Stress management:</strong> Meditation, yoga, hobbies</li>
                        <li><strong>No smoking:</strong> Smoking increases risk by 30-40%</li>
                        <li><strong>Moderate alcohol:</strong> If consumed, limit to 1 drink/day (women), 2 drinks/day (men)</li>
                    </ul>
                    
                    <div class="alert alert-success mt-4">
                        <h5><i class="fas fa-lightbulb me-2"></i>Success Stories</h5>
                        <p>The Diabetes Prevention Program study showed that lifestyle intervention reduced diabetes incidence by 58% over 3 years compared to placebo.</p>
                    </div>
                `
            },
            3: {
                title: "Early Warning Signs of Diabetes",
                content: `
                    <h4>Recognizing Diabetes Symptoms</h4>
                    <p>Early detection of diabetes symptoms can prevent serious complications. Many people have prediabetes or early diabetes without noticeable symptoms.</p>
                    
                    <h5 class="mt-4">Common Symptoms (The "3 Ps"):</h5>
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <i class="fas fa-tint fa-3x text-primary mb-3"></i>
                                    <h5>Polyuria</h5>
                                    <p>Frequent urination (especially at night)</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <i class="fas fa-glass-whiskey fa-3x text-primary mb-3"></i>
                                    <h5>Polydipsia</h5>
                                    <p>Excessive thirst</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <i class="fas fa-utensils fa-3x text-primary mb-3"></i>
                                    <h5>Polyphagia</h5>
                                    <p>Increased hunger</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">Other Important Symptoms:</h5>
                    <div class="table-responsive">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Symptom</th>
                                    <th>Description</th>
                                    <th>Why It Happens</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Fatigue</strong></td>
                                    <td>Extreme tiredness, lack of energy</td>
                                    <td>Cells aren't getting enough glucose for energy</td>
                                </tr>
                                <tr>
                                    <td><strong>Blurred Vision</strong></td>
                                    <td>Difficulty focusing, fluctuating eyesight</td>
                                    <td>High blood sugar affects eye lens shape</td>
                                </tr>
                                <tr>
                                    <td><strong>Slow Healing</strong></td>
                                    <td>Cuts/bruises take longer to heal</td>
                                    <td>High glucose impairs immune function</td>
                                </tr>
                                <tr>
                                    <td><strong>Tingling/Numbness</strong></td>
                                    <td>In hands, feet, or legs</td>
                                    <td>Nerve damage (neuropathy)</td>
                                </tr>
                                <tr>
                                    <td><strong>Unexplained Weight Loss</strong></td>
                                    <td>Despite eating normally</td>
                                    <td>Body burns fat/muscle for energy</td>
                                </tr>
                                <tr>
                                    <td><strong>Recurrent Infections</strong></td>
                                    <td>Skin, gum, or urinary infections</td>
                                    <td>High glucose supports bacterial/fungal growth</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <h5 class="mt-4">When to See a Doctor:</h5>
                    <div class="alert alert-warning">
                        <p>Consult a healthcare provider if you experience:</p>
                        <ul>
                            <li>Any combination of the above symptoms</li>
                            <li>Family history of diabetes</li>
                            <li>Over 45 years old with risk factors</li>
                            <li>Previous gestational diabetes</li>
                            <li>High blood pressure or cholesterol</li>
                        </ul>
                    </div>
                    
                    <h5 class="mt-4">Diagnostic Tests:</h5>
                    <ul>
                        <li><strong>Fasting Blood Glucose:</strong> ≥126 mg/dL indicates diabetes</li>
                        <li><strong>HbA1c Test:</strong> ≥6.5% indicates diabetes</li>
                        <li><strong>Oral Glucose Tolerance Test:</strong> ≥200 mg/dL after 2 hours</li>
                        <li><strong>Random Blood Glucose:</strong> ≥200 mg/dL with symptoms</li>
                    </ul>
                `
            },
            4: {
                title: "Diabetes-Friendly Diet Guide",
                content: `
                    <h4>Nutrition for Diabetes Management</h4>
                    <p>A balanced diet is the cornerstone of diabetes management. The goal is to maintain stable blood sugar levels while getting proper nutrition.</p>
                    
                    <h5 class="mt-4">Plate Method (Recommended by ADA):</h5>
                    <div class="text-center mb-4">
                        <div style="width: 300px; height: 300px; border-radius: 50%; background: conic-gradient(#2ecc71 50%, #3498db 25%, #e74c3c 25%); display: inline-block; position: relative;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; width: 100px; height: 100px; border-radius: 50%;"></div>
                        </div>
                    </div>
                    <div class="row text-center">
                        <div class="col-md-4">
                            <span class="badge bg-success p-2" style="font-size: 1rem;">50% Non-Starchy Vegetables</span>
                            <p class="small mt-2">Broccoli, spinach, carrots, peppers, tomatoes</p>
                        </div>
                        <div class="col-md-4">
                            <span class="badge bg-primary p-2" style="font-size: 1rem;">25% Lean Protein</span>
                            <p class="small mt-2">Chicken, fish, tofu, legumes, eggs</p>
                        </div>
                        <div class="col-md-4">
                            <span class="badge bg-danger p-2" style="font-size: 1rem;">25% Whole Grains/Starchy Vegetables</span>
                            <p class="small mt-2">Brown rice, quinoa, sweet potatoes, beans</p>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">Foods to Include:</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card border-success mb-3">
                                <div class="card-header bg-success text-white">
                                    <h6 class="mb-0"><i class="fas fa-check me-2"></i>Best Choices</h6>
                                </div>
                                <div class="card-body">
                                    <ul>
                                        <li><strong>Vegetables:</strong> Leafy greens, broccoli, cauliflower</li>
                                        <li><strong>Fruits:</strong> Berries, apples, oranges (in moderation)</li>
                                        <li><strong>Whole grains:</strong> Oats, quinoa, whole wheat</li>
                                        <li><strong>Proteins:</strong> Fish, skinless poultry, legumes</li>
                                        <li><strong>Healthy fats:</strong> Avocado, nuts, olive oil</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card border-danger mb-3">
                                <div class="card-header bg-danger text-white">
                                    <h6 class="mb-0"><i class="fas fa-times me-2"></i>Limit or Avoid</h6>
                                </div>
                                <div class="card-body">
                                    <ul>
                                        <li><strong>Sugary drinks:</strong> Sodas, sweet tea, juice</li>
                                        <li><strong>Processed carbs:</strong> White bread, pastries</li>
                                        <li><strong>Fried foods:</strong> High in unhealthy fats</li>
                                        <li><strong>Processed meats:</strong> Bacon, sausages</li>
                                        <li><strong>High-sugar snacks:</strong> Candy, cookies</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">Glycemic Index (GI) Guide:</h5>
                    <p>Choose low-GI foods for slower glucose absorption:</p>
                    <div class="table-responsive">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Low GI (< 55)</th>
                                    <th>Medium GI (56-69)</th>
                                    <th>High GI (> 70)</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="table-success">Most fruits & vegetables</td>
                                    <td class="table-warning">Whole wheat bread</td>
                                    <td class="table-danger">White bread</td>
                                </tr>
                                <tr>
                                    <td class="table-success">Legumes</td>
                                    <td class="table-warning">Brown rice</td>
                                    <td class="table-danger">White rice</td>
                                </tr>
                                <tr>
                                    <td class="table-success">Sweet potatoes</td>
                                    <td class="table-warning">Oatmeal</td>
                                    <td class="table-danger">Corn flakes</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <h5 class="mt-4">Meal Timing Tips:</h5>
                    <ul>
                        <li>Eat regular meals (3 main meals + 1-2 snacks if needed)</li>
                        <li>Don't skip breakfast</li>
                        <li>Space meals 4-5 hours apart</li>
                        <li>Monitor blood sugar before/after meals</li>
                        <li>Stay hydrated with water</li>
                    </ul>
                `
            },
            5: {
                title: "Modern Diabetes Treatments",
                content: `
                    <h4>Advances in Diabetes Treatment</h4>
                    <p>Diabetes management has evolved from basic insulin injections to sophisticated technology-driven solutions.</p>
                    
                    <h5 class="mt-4">1. Medications:</h5>
                    <div class="table-responsive">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Medication Class</th>
                                    <th>How It Works</th>
                                    <th>Examples</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Biguanides</strong></td>
                                    <td>Decreases liver glucose production</td>
                                    <td>Metformin (most common first-line)</td>
                                </tr>
                                <tr>
                                    <td><strong>GLP-1 Receptor Agonists</strong></td>
                                    <td>Increases insulin, decreases glucagon, slows gastric emptying</td>
                                    <td>Liraglutide, Semaglutide</td>
                                </tr>
                                <tr>
                                    <td><strong>SGLT2 Inhibitors</strong></td>
                                    <td>Blocks glucose reabsorption in kidneys</td>
                                    <td>Empagliflozin, Canagliflozin</td>
                                </tr>
                                <tr>
                                    <td><strong>DPP-4 Inhibitors</strong></td>
                                    <td>Prolongs action of incretin hormones</td>
                                    <td>Sitagliptin, Linagliptin</td>
                                </tr>
                                <tr>
                                    <td><strong>Sulfonylureas</strong></td>
                                    <td>Stimulates insulin release</td>
                                    <td>Glibenclamide, Glimepiride</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <h5 class="mt-4">2. Insulin Therapy:</h5>
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <h6>Rapid-Acting</h6>
                                    <p>Meal-time insulin</p>
                                    <small>Onset: 15 min</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <h6>Long-Acting</h6>
                                    <p>Basal insulin</p>
                                    <small>Lasts 24+ hours</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <h6>Premixed</h6>
                                    <p>Combination</p>
                                    <small>For convenience</small>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">3. Technology Advances:</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-header bg-primary text-white">
                                    <h6 class="mb-0"><i class="fas fa-syringe me-2"></i>Insulin Pumps</h6>
                                </div>
                                <div class="card-body">
                                    <p>Small devices that deliver continuous insulin:</p>
                                    <ul>
                                        <li>More precise dosing</li>
                                        <li>Fewer injections</li>
                                        <li>Better HbA1c control</li>
                                        <li>Hybrid closed-loop systems available</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-header bg-success text-white">
                                    <h6 class="mb-0"><i class="fas fa-chart-line me-2"></i>CGM Systems</h6>
                                </div>
                                <div class="card-body">
                                    <p>Continuous Glucose Monitors:</p>
                                    <ul>
                                        <li>Real-time glucose readings</li>
                                        <li>Trend arrows and alerts</li>
                                        <li>Reduces fingersticks</li>
                                        <li>Data sharing capabilities</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">4. Artificial Pancreas Systems:</h5>
                    <div class="alert alert-info">
                        <p>Closed-loop systems that automatically adjust insulin delivery based on CGM readings. Considered the future of type 1 diabetes management.</p>
                    </div>
                    
                    <h5 class="mt-4">5. Future Treatments:</h5>
                    <ul>
                        <li><strong>Stem cell therapy:</strong> Regenerating insulin-producing cells</li>
                        <li><strong>Smart insulin:</strong> Insulin that activates only when needed</li>
                        <li><strong>Gene therapy:</strong> Correcting genetic defects</li>
                        <li><strong>Oral insulin:</strong> Non-injectable insulin formulations</li>
                    </ul>
                `
            },
            6: {
                title: "Latest Diabetes Research",
                content: `
                    <h4>Cutting-Edge Diabetes Research</h4>
                    <p>Scientific advances are transforming our understanding and treatment of diabetes.</p>
                    
                    <h5 class="mt-4">1. Precision Medicine:</h5>
                    <p>Tailoring treatment based on individual characteristics:</p>
                    <ul>
                        <li>Genetic profiling for medication response</li>
                        <li>Personalized nutrition plans based on microbiome</li>
                        <li>Wearable technology integration</li>
                        <li>AI-powered treatment recommendations</li>
                    </ul>
                    
                    <h5 class="mt-4">2. Immunotherapy for Type 1 Diabetes:</h5>
                    <div class="card mb-3">
                        <div class="card-body">
                            <p>New approaches to halt autoimmune destruction:</p>
                            <ul>
                                <li><strong>Teplizumab:</strong> First drug to delay type 1 diabetes onset (approved 2022)</li>
                                <li><strong>BCG vaccine:</strong> Shows promise in reversing advanced type 1 diabetes</li>
                                <li><strong>Stem cell-derived beta cells:</strong> Clinical trials showing promising results</li>
                            </ul>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">3. Gut Microbiome Research:</h5>
                    <p>The gut bacteria connection to diabetes:</p>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h6>Key Findings:</h6>
                                    <ul>
                                        <li>Specific bacteria linked to insulin resistance</li>
                                        <li>Fiber-rich diets promote beneficial bacteria</li>
                                        <li>Probiotics may improve glucose control</li>
                                        <li>Fecal transplants being studied</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h6>Practical Applications:</h6>
                                    <ul>
                                        <li>Personalized probiotic recommendations</li>
                                        <li>Diet based on microbiome analysis</li>
                                        <li>Microbiome as treatment target</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">4. Digital Health Revolution:</h5>
                    <div class="row">
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <i class="fas fa-mobile-alt fa-2x text-primary mb-2"></i>
                                    <h6>Mobile Apps</h6>
                                    <p class="small">Carb counting, medication tracking, education</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <i class="fas fa-robot fa-2x text-primary mb-2"></i>
                                    <h6>AI Coaches</h6>
                                    <p class="small">Personalized advice, predictive alerts</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center mb-3">
                                <div class="card-body">
                                    <i class="fas fa-cloud fa-2x text-primary mb-2"></i>
                                    <h6>Telemedicine</h6>
                                    <p class="small">Remote monitoring, virtual consultations</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h5 class="mt-4">5. Promising Clinical Trials:</h5>
                    <div class="table-responsive">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Treatment</th>
                                    <th>Phase</th>
                                    <th>Potential Impact</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Oral GLP-1 agonists</td>
                                    <td>Phase 3</td>
                                    <td>Non-injectable weight loss/diabetes drugs</td>
                                </tr>
                                <tr>
                                    <td>Implantable CGM sensors</td>
                                    <td>Phase 2</td>
                                    <td>6-month continuous monitoring</td>
                                </tr>
                                <tr>
                                    <td>Smart contact lenses</td>
                                    <td>Phase 1</td>
                                    <td>Glucose monitoring via tears</td>
                                </tr>
                                <tr>
                                    <td>Beta cell encapsulation</td>
                                    <td>Pre-clinical</td>
                                    <td>Protect transplanted cells from immune attack</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="alert alert-success mt-4">
                        <h5><i class="fas fa-bullhorn me-2"></i>The Future is Bright</h5>
                        <p>With rapid advancements in technology and medicine, we're moving toward a future where diabetes can be effectively managed, prevented, and potentially cured.</p>
                    </div>
                `
            }
        };
        
        // Show article in modal
        function showArticle(id) {
            if (articles[id]) {
                document.getElementById('articleModalTitle').textContent = articles[id].title;
                document.getElementById('articleModalContent').innerHTML = articles[id].content;
                const modal = new bootstrap.Modal(document.getElementById('articleModal'));
                modal.show();
            }
        }
    </script>
</body>
</html>"""

# Save the templates
with open("templates/home.html", "w", encoding="utf-8") as f:
    f.write(home_html)

with open("templates/calculator.html", "w", encoding="utf-8") as f:
    f.write(calculator_html)

with open("templates/articles.html", "w", encoding="utf-8") as f:
    f.write(articles_html)

# Route definitions
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Home page"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/calculator", response_class=HTMLResponse)
async def calculator_page(request: Request):
    """Calculator page"""
    return templates.TemplateResponse("calculator.html", {"request": request})

@app.get("/articles", response_class=HTMLResponse)
async def articles_page(request: Request):
    """Articles page"""
    return templates.TemplateResponse("articles.html", {"request": request})

# Prediction endpoint (same as before)
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    gender: str = Form(...),
    age: float = Form(...),
    bmi: float = Form(...),
    smoking_history: str = Form(...),
    hbA1c_level: float = Form(...),
    blood_glucose_level: float = Form(...),
    glucose_unit: str = Form("mgdl")
):
    """Process prediction"""
    try:
        # Convert glucose units
        blood_glucose_mgdl = convert_glucose(blood_glucose_level, glucose_unit)
        
        # Prepare input
        user_input = {
            "gender": gender,
            "age": age,
            "bmi": bmi,
            "smoking_history": smoking_history,
            "hbA1c_level": hbA1c_level,
            "blood_glucose_level": blood_glucose_mgdl
        }
        
        # Get prediction
        result = predictor.predict_diabetes(user_input)
        
        if result.get('error'):
            error_html = f"""
            <div class="container mt-5" style="padding-top: 100px;">
                <div class="alert alert-danger">
                    <h4><i class="fas fa-exclamation-triangle me-2"></i>Error</h4>
                    <p>{result.get('message')}</p>
                    <a href="/calculator" class="btn btn-primary">Try Again</a>
                </div>
            </div>
            """
            return HTMLResponse(content=error_html)
        
        # Extract results
        prediction = result['prediction_label']
        probability = result['probability']
        confidence = result['confidence']
        advice = result['medical_advice']
        
        # Generate risk meter
        risk_meter = generate_risk_meter(probability)
        risk_color = get_risk_color(probability)
        
        # Get BMI category
        bmi_category, bmi_color, bmi_desc = get_bmi_category(bmi)
        
        # Generate chart data
        chart_data = json.dumps({
            "datasets": [{
                "data": [probability * 100, (1 - probability) * 100],
                "backgroundColor": [
                    "#dc3545" if probability > 0.6 else "#ffc107" if probability > 0.3 else "#28a745",
                    "#e9ecef"
                ]
            }],
            "labels": ["Diabetes Risk", "Low Risk"]
        })
        
        # Generate results HTML
        results_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results - Diabetes Risk Assessment</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        .navbar {{
            background: rgba(44, 62, 80, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }}
        
        .card {{
            border-radius: 15px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            border: none;
            margin-bottom: 20px;
        }}
        
        .risk-badge {{
            font-size: 1.2em;
            padding: 10px 20px;
            border-radius: 20px;
        }}
        
        .result-card {{
            animation: fadeIn 0.8s ease;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-heartbeat me-2"></i>
                <strong>DiaRisk AI</strong>
            </a>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/calculator">Calculator</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/articles">Articles</a>
                    </li>
                </ul>
            </div>
            <a href="/calculator" class="btn btn-outline-light">
                <i class="fas fa-redo me-1"></i>New Assessment
            </a>
        </div>
    </nav>
    
    <div class="container mt-5" style="padding-top: 100px;">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <!-- Header -->
                <div class="text-center mb-5 text-white">
                    <h1 class="mb-3">Diabetes Risk Assessment Results</h1>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
                </div>
                
                <!-- Summary Card -->
                <div class="card result-card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h3 class="mb-0"><i class="fas fa-chart-line me-2"></i>Risk Summary</h3>
                    </div>
                    <div class="card-body">
                        <div class="row align-items-center">
                            <div class="col-md-6">
                                <div class="probability-display text-{risk_color} mb-3 text-center" style="font-size: 3rem; font-weight: bold;">
                                    {probability:.1%}
                                </div>
                                <h4 class="text-center mb-4">
                                    <span class="badge risk-badge bg-{risk_color}">
                                        {advice['risk_level']} RISK
                                    </span>
                                </h4>
                                <p class="text-center fs-5">
                                    Prediction: <strong>{prediction}</strong><br>
                                    <small class="text-muted">Confidence: {confidence:.1%}</small>
                                </p>
                            </div>
                            <div class="col-md-6">
                                <canvas id="riskChart" height="200"></canvas>
                            </div>
                        </div>
                        
                        <!-- Risk Meter -->
                        <div class="mt-4">
                            <label class="form-label">Risk Level:</label>
                            {risk_meter}
                            <div class="d-flex justify-content-between mt-2">
                                <small>Low Risk (0-30%)</small>
                                <small>Moderate Risk (30-60%)</small>
                                <small>High Risk (60-100%)</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Medical Advice -->
                <div class="row">
                    <div class="col-md-7">
                        <div class="card h-100">
                            <div class="card-header bg-success text-white">
                                <h4 class="mb-0"><i class="fas fa-stethoscope me-2"></i>Medical Recommendations</h4>
                            </div>
                            <div class="card-body">
                                <div class="border-start border-5 border-{risk_color} p-3 mb-3">
                                    <h5><i class="fas fa-bullhorn me-2"></i>Immediate Actions</h5>
                                    <ul>
                                        {"".join(f'<li>{rec}</li>' for rec in advice['recommendations'])}
                                    </ul>
                                </div>
                                
                                {f'<div class="alert alert-warning"><h5><i class="fas fa-exclamation-triangle me-2"></i>Warnings</h5><ul>{"".join(f"<li>{w}</li>" for w in advice["warnings"])}</ul></div>' if advice['warnings'] else ''}
                                
                                <div class="card bg-light">
                                    <div class="card-body">
                                        <h5><i class="fas fa-footprints me-2"></i>Next Steps</h5>
                                        <ul>
                                            {"".join(f'<li>{step}</li>' for step in advice['next_steps'])}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Lifestyle Tips -->
                    <div class="col-md-5">
                        <div class="card h-100">
                            <div class="card-header bg-info text-white">
                                <h4 class="mb-0"><i class="fas fa-heart me-2"></i>Lifestyle Tips</h4>
                            </div>
                            <div class="card-body">
                                <ul class="list-group list-group-flush">
                                    {"".join(f'<li class="list-group-item"><i class="fas fa-check-circle text-success me-2"></i>{tip}</li>' for tip in advice['lifestyle_tips'])}
                                </ul>
                                
                                <div class="mt-4">
                                    <h6><i class="fas fa-chart-bar me-2"></i>Parameter Analysis</h6>
                                    <div class="table-responsive">
                                        <table class="table table-sm">
                                            <tbody>
                                                <tr>
                                                    <td><strong>BMI</strong></td>
                                                    <td>{bmi:.1f}</td>
                                                    <td><span class="badge bg-{bmi_color}" title="{bmi_desc}">{bmi_category}</span></td>
                                                </tr>
                                                <tr>
                                                    <td><strong>HbA1c</strong></td>
                                                    <td>{hbA1c_level:.1f}%</td>
                                                    <td>
                                                        <span class="badge bg-{'success' if hbA1c_level < 5.7 else 'warning' if hbA1c_level < 6.5 else 'danger'}">
                                                            {{'Normal' if hbA1c_level < 5.7 else 'Prediabetes' if hbA1c_level < 6.5 else 'Diabetes'}}
                                                        </span>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Glucose</strong></td>
                                                    <td>{blood_glucose_mgdl:.0f} mg/dL</td>
                                                    <td>
                                                        <span class="badge bg-{'success' if blood_glucose_mgdl < 100 else 'warning' if blood_glucose_mgdl < 126 else 'danger'}">
                                                            {{'Normal' if blood_glucose_mgdl < 100 else 'Prediabetes' if blood_glucose_mgdl < 126 else 'Diabetes'}}
                                                        </span>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Age</strong></td>
                                                    <td>{age:.0f} years</td>
                                                    <td>
                                                        <span class="badge bg-{'success' if age < 45 else 'warning' if age < 65 else 'danger'}">
                                                            {{'Young' if age < 45 else 'Middle' if age < 65 else 'Senior'}}
                                                        </span>
                                                    </td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Disclaimer -->
                <div class="alert alert-warning mt-4">
                    <h5><i class="fas fa-exclamation-circle me-2"></i>Important Disclaimer</h5>
                    <p class="mb-0">
                        This assessment is for screening and educational purposes only. It is not a substitute for 
                        professional medical advice, diagnosis, or treatment. Always consult with healthcare professionals.
                    </p>
                </div>
                
               <!-- Action Buttons -->
<div class="d-grid gap-2 d-md-flex justify-content-md-center mt-4">
    <a href="/calculator" class="btn btn-primary btn-lg me-2">
        <i class="fas fa-redo me-2"></i>New Assessment
    </a>
    <button class="btn btn-primary btn-lg me-2" onclick="window.print()">
        <i class="fas fa-print me-2"></i>Print Results
    </button>
    <a href="/articles" class="btn btn-success btn-lg ms-2">
        <i class="fas fa-book me-2"></i>Learn More
    </a>
</div>

            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Risk Chart
        const ctx = document.getElementById('riskChart').getContext('2d');
        const chartData = {chart_data};
        
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: chartData.labels,
                datasets: chartData.datasets
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return `${{context.label}}: ${{context.parsed}}%`;
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return HTMLResponse(content=results_html)
        
    except Exception as e:
        error_html = f"""
        <div class="container mt-5" style="padding-top: 100px;">
            <div class="alert alert-danger">
                <h4><i class="fas fa-exclamation-triangle me-2"></i>System Error</h4>
                <p>An error occurred: {str(e)}</p>
                <a href="/calculator" class="btn btn-primary">Return to Calculator</a>
            </div>
        </div>
        """
        return HTMLResponse(content=error_html)

# JSON API endpoint
@app.post("/predict_json")
def predict_json(data: dict):
    """JSON API endpoint"""
    try:
        result = predictor.predict_diabetes(data)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "error": True,
            "message": str(e)
        })

# Health check
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
