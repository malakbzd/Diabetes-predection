import pandas as pd
import numpy as np
import pickle
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self, model_path="diabetes_model.pkl"):
        """
        Initialize Diabetes Predictor
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.encoder = {}
        self.feature_names = None
        self.class_names = ['No Diabetes', 'Diabetes']
        
        # Define acceptable ranges for validation
        self.validation_ranges = {
            'age': (0, 120),
            'bmi': (10, 60),
            'hbA1c_level': (3, 15),
            'blood_glucose_level': (50, 300)
        }
        
        # Medical insights
        self.medical_info = {
            'bmi': {
                'name': 'Body Mass Index',
                'categories': {
                    'Underweight': '<18.5',
                    'Normal': '18.5-24.9',
                    'Overweight': '25-29.9',
                    'Obese': '30+'
                }
            },
            'hbA1c_level': {
                'name': 'Hemoglobin A1c',
                'categories': {
                    'Normal': '<5.7%',
                    'Prediabetes': '5.7-6.4%',
                    'Diabetes': '≥6.5%'
                }
            }
        }

    def load_and_preprocess_data(self, filepath="diabetes_dataset.csv"):
        """
        Load and preprocess the diabetes dataset
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset '{filepath}' not found.")
            
            print(f"📁 Loading dataset from {filepath}")
            df = pd.read_csv(filepath)
            
            print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Check for diabetes column
            if 'diabetes' not in df.columns:
                raise ValueError("Dataset must contain 'diabetes' column")
            
            # Select only the columns we need
            # Your dataset has: gender, age, smoking_history, bmi, hbA1c_level, blood_glucose_level
            required_cols = ['gender', 'age', 'smoking_history', 'bmi', 'hbA1c_level', 'blood_glucose_level', 'diabetes']
            
            # Check which columns exist in the dataset
            available_cols = []
            for col in required_cols:
                if col in df.columns:
                    available_cols.append(col)
                else:
                    print(f"⚠ Warning: Column '{col}' not found in dataset")
            
            print(f"🔍 Using columns: {available_cols}")
            df_clean = df[available_cols].copy()
            
            # Handle missing values
            print("\n🔍 Checking for missing values...")
            missing_values = df_clean.isnull().sum()
            if missing_values.any():
                print(f"⚠ Missing values found:\n{missing_values[missing_values > 0]}")
                # Fill numeric columns with median
                for col in ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']:
                    if col in df_clean.columns and df_clean[col].isnull().any():
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
            else:
                print("✅ No missing values found")
            
            # Encode categorical variables
            print("\n🔧 Encoding categorical variables...")
            for col in ['gender', 'smoking_history']:
                if col in df_clean.columns:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                    self.encoder[col] = le
                    print(f"   → Encoded {col}: {list(le.classes_)}")
            
            # Validate data ranges
            print("\n⚕️ Validating medical ranges...")
            for col, (min_val, max_val) in self.validation_ranges.items():
                if col in df_clean.columns:
                    outliers = df_clean[(df_clean[col] < min_val) | (df_clean[col] > max_val)]
                    if len(outliers) > 0:
                        print(f"   ⚠ {col}: {len(outliers)} values outside range {min_val}-{max_val}")
            
            # Define feature names
            self.feature_names = [col for col in df_clean.columns if col != 'diabetes']
            
            # Diabetes distribution
            diabetes_counts = df_clean['diabetes'].value_counts()
            print(f"\n📊 Diabetes Distribution:")
            print(f"   → Non-diabetic: {diabetes_counts.get(0, 0)} ({diabetes_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
            print(f"   → Diabetic: {diabetes_counts.get(1, 0)} ({diabetes_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
            
            print("\n✅ Data preprocessing completed successfully")
            return df_clean
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise

    def train_model(self, df):
        """
        Train Random Forest Classifier
        """
        print("\n🤖 Training Diabetes Prediction Model...")
        
        # Prepare features and target
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"   → Features used: {self.feature_names}")
        print(f"   → Total samples: {len(df)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   → Training samples: {X_train.shape[0]}")
        print(f"   → Testing samples: {X_test.shape[0]}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        print("   → Training Random Forest Classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"   → Training Accuracy: {train_acc:.3f}")
        print(f"   → Testing Accuracy: {test_acc:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📈 Feature Importance:")
        for idx, row in feature_importance.iterrows():
            print(f"   → {row['feature']}: {row['importance']:.3f}")
        
        print("\n✅ Model training completed!")
        return self.model

    def save_model(self):
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'encoder': self.encoder,
                'feature_names': self.feature_names,
                'class_names': self.class_names
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self):
        """Load model from disk"""
        try:
            if not os.path.exists(self.model_path):
                print(f"📭 No saved model found at {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.encoder = model_data['encoder']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data['class_names']
            
            print(f"📂 Model loaded from {self.model_path}")
            print(f"   → Features: {len(self.feature_names)}")
            print(f"   → Model type: {type(self.model).__name__}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def validate_input(self, user_input):
        """
        Validate user input
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['gender', 'age', 'bmi', 'smoking_history', 'hbA1c_level', 'blood_glucose_level']
        for field in required_fields:
            if field not in user_input:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors, warnings
        
        # Validate numeric ranges
        for field, (min_val, max_val) in self.validation_ranges.items():
            if field in user_input:
                try:
                    value = float(user_input[field])
                    if value < min_val:
                        errors.append(f"{field}: {value} is below minimum {min_val}")
                    elif value > max_val:
                        errors.append(f"{field}: {value} is above maximum {max_val}")
                    
                    # Medical warnings
                    if field == 'bmi':
                        if value < 18.5:
                            warnings.append(f"BMI {value:.1f}: Underweight")
                        elif value >= 25:
                            warnings.append(f"BMI {value:.1f}: Overweight/Obese")
                    
                    elif field == 'hbA1c_level':
                        if value >= 6.5:
                            warnings.append(f"HbA1c {value:.1f}%: Diabetes range")
                        elif value >= 5.7:
                            warnings.append(f"HbA1c {value:.1f}%: Prediabetes range")
                    
                    elif field == 'blood_glucose_level':
                        if value >= 126:
                            warnings.append(f"Blood glucose {value:.0f} mg/dL: Diabetes range")
                        elif value >= 100:
                            warnings.append(f"Blood glucose {value:.0f} mg/dL: Prediabetes range")
                
                except ValueError:
                    errors.append(f"{field}: '{user_input[field]}' is not a valid number")
        
        return len(errors) == 0, errors, warnings

    def get_medical_advice(self, prediction_result, user_input):
        """
        Generate personalized medical advice
        """
        advice = {
            'risk_level': prediction_result['risk_level'],
            'recommendations': [],
            'warnings': [],
            'next_steps': [],
            'lifestyle_tips': []
        }
        
        risk_score = prediction_result.get('probability', 0.5)
        
        # Risk-based recommendations
        if risk_score > 0.7:
            advice['risk_level'] = 'HIGH'
            advice['recommendations'].append("Immediate medical consultation recommended")
            advice['recommendations'].append("Schedule fasting blood glucose and HbA1c tests")
            advice['warnings'].append("High diabetes risk detected")
        elif risk_score > 0.4:
            advice['risk_level'] = 'MODERATE'
            advice['recommendations'].append("Regular monitoring recommended")
            advice['recommendations'].append("Follow-up test in 3-6 months")
        else:
            advice['risk_level'] = 'LOW'
            advice['recommendations'].append("Maintain healthy lifestyle")
            advice['recommendations'].append("Annual check-up recommended")
        
        # Parameter-specific advice
        bmi = float(user_input.get('bmi', 25))
        hba1c = float(user_input.get('hbA1c_level', 5.5))
        glucose = float(user_input.get('blood_glucose_level', 100))
        
        # BMI advice
        if bmi >= 30:
            advice['lifestyle_tips'].append("Weight management: Aim for 5-10% weight loss")
            advice['lifestyle_tips'].append("Consult nutritionist for meal planning")
        elif bmi >= 25:
            advice['lifestyle_tips'].append("Moderate weight loss can reduce diabetes risk by 50%")
        
        # HbA1c advice
        if hba1c >= 6.5:
            advice['warnings'].append(f"HbA1c {hba1c:.1f}% indicates diabetes")
            advice['next_steps'].append("Consult endocrinologist for treatment plan")
        elif hba1c >= 5.7:
            advice['warnings'].append(f"HbA1c {hba1c:.1f}% indicates prediabetes")
            advice['next_steps'].append("Participate in diabetes prevention program")
        
        # General lifestyle tips
        advice['lifestyle_tips'].append("150 minutes of moderate exercise weekly")
        advice['lifestyle_tips'].append("Increase fiber intake (vegetables, whole grains)")
        advice['lifestyle_tips'].append("Avoid smoking and limit alcohol")
        
        return advice

    def predict_diabetes(self, user_input):
        """
        Main prediction method
        """
        try:
            # Validate input
            is_valid, errors, warnings = self.validate_input(user_input)
            if not is_valid:
                return {
                    'error': True,
                    'message': 'Input validation failed',
                    'errors': errors
                }
            
            if self.model is None or self.scaler is None:
                return {"error": True, "message": "Model not trained or loaded."}
            
            # Prepare input data
            input_df = pd.DataFrame([user_input])
            
            # Encode categorical variables
            for col, encoder in self.encoder.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col].astype(str))
                    except ValueError:
                        # If label not seen during training, use default
                        input_df[col] = 0
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns
            input_df = input_df[self.feature_names]
            
            # Scale features
            scaled_features = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            # Get medical advice
            medical_advice = self.get_medical_advice({
                'prediction': prediction,
                'probability': probabilities[1],
                'risk_level': 'HIGH' if probabilities[1] > 0.7 else 'MODERATE' if probabilities[1] > 0.4 else 'LOW'
            }, user_input)
            
            # Prepare response
            result = {
                'error': False,
                'prediction': int(prediction),
                'prediction_label': self.class_names[prediction],
                'probability': float(probabilities[1]),
                'confidence': float(max(probabilities)),
                'medical_advice': medical_advice,
                'warnings': warnings,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Prediction error: {str(e)}'
            }


def main():
    """
    Command-line interface
    """
    print("=" * 60)
    print("🩺 DIABETES RISK PREDICTION SYSTEM")
    print("=" * 60)
    
    predictor = DiabetesPredictor()
    
    # Try to load existing model
    if not predictor.load_model():
        print("\n🔧 No saved model found. Training new model...")
        try:
            df = predictor.load_and_preprocess_data("diabetes_dataset.csv")
            predictor.train_model(df)
            predictor.save_model()
            print("✅ Model trained and saved successfully!")
        except Exception as e:
            print(f"❌ Failed to train model: {e}")
            return
    
    print("\n📝 Please enter your health information:")
    print("   (Press Ctrl+C to exit)\n")
    
    try:
        user_input = {
            "gender": input("Gender (Female/Male/Other): ").strip().title(),
            "age": input("Age (0-120): ").strip(),
            "bmi": input("BMI (10-60, e.g., 24.5): ").strip(),
            "smoking_history": input(
                "Smoking History (never/not current/current/no info/ever/former): "
            ).strip().lower(),
            "hbA1c_level": input("HbA1c Level (3-15, e.g., 5.7): ").strip(),
            "blood_glucose_level": input("Blood Glucose Level (50-300 mg/dL): ").strip(),
        }
        
        print("\n" + "=" * 60)
        print("🔍 Analyzing your data...")
        
        response = predictor.predict_diabetes(user_input)
        
        if response.get('error'):
            print(f"❌ Error: {response.get('message')}")
            return
        
        print("\n" + "=" * 60)
        print("📊 PREDICTION RESULTS")
        print("=" * 60)
        
        print(f"\n🧪 Diagnosis: {response['prediction_label']}")
        print(f"📈 Risk Probability: {response['probability']:.1%}")
        
        advice = response['medical_advice']
        print(f"\n⚕️ Risk Level: {advice['risk_level']}")
        
        if advice['recommendations']:
            print("\n📋 Recommendations:")
            for rec in advice['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "=" * 60)
        print("🩺 Remember: This is a screening tool, not a diagnosis.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n👋 Exiting... Stay healthy!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()