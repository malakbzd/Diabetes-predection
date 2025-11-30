# predictor.py - Your exact diabetes prediction logic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import json
import os
from datetime import datetime

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_medians = None
        self.is_trained = False

        self.zero_invalid_cols = ['bmi', 'hbA1c_level', 'blood_glucose_level']
        self.gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
        self.smoking_mapping = {
            'never': 0, 'not current': 1, 'current': 2,
            'No Info': 3, 'ever': 4, 'former': 5
        }

    def load_and_preprocess_data(self, file_path="diabetes.csv"):
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded dataset with {len(df)} records")

            df = self._clean_data(df)

            if "diabetes" not in df.columns:
                raise ValueError("❌ Dataset missing 'diabetes' column")
                
            return df

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

    def _clean_data(self, df):
        self.training_medians = {}

        for col in self.zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.training_medians[col] = float(median_val)

        df["gender"] = df["gender"].map(self.gender_mapping).fillna(self.gender_mapping["Other"])
        df["smoking_history"] = df["smoking_history"].map(self.smoking_mapping).fillna(
            self.smoking_mapping["No Info"]
        )

        return df

    def train_model(self, df, test_size=0.2, random_state=42):
        try:
            X = df.drop("diabetes", axis=1)
            y = df["diabetes"]
            self.feature_names = list(X.columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None]
            }

            grid = GridSearchCV(
                RandomForestClassifier(random_state=random_state),
                param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1
            )
            grid.fit(X_train_scaled, y_train)

            self.model = grid.best_estimator_
            self.is_trained = True

            y_pred = self.model.predict(X_test_scaled)
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
            
            print(f"📊 Model trained - Accuracy: {accuracy_score(y_test, y_pred):.1%}")

        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

    def save_model(self):
        if not self.is_trained:
            raise ValueError("Train the model first!")
            
        pickle.dump(self.model, open("diabetes_model.pkl", "wb"))
        pickle.dump(self.scaler, open("scaler.pkl", "wb"))
        
        metadata = {
            "feature_names": self.feature_names,
            "training_medians": self.training_medians,
            "timestamp": datetime.now().isoformat()
        }
        
        json.dump(metadata, open("model_metadata.json", "w"), indent=2)
        print("💾 Model saved!")

    def load_model(self):
        if not (os.path.exists("diabetes_model.pkl") and os.path.exists("scaler.pkl")):
            return False
            
        self.model = pickle.load(open("diabetes_model.pkl", "rb"))
        self.scaler = pickle.load(open("scaler.pkl", "rb"))
        
        if os.path.exists("model_metadata.json"):
            meta = json.load(open("model_metadata.json"))
            self.feature_names = meta.get("feature_names", [])
            self.training_medians = meta.get("training_medians", {})
        
        self.is_trained = True
        return True

    def predict_diabetes(self, user_input: dict) -> dict:
        if not self.is_trained:
            return {"error": "Model not ready"}

        try:
            df = pd.DataFrame([user_input])
            
            df["gender"] = df["gender"].map(self.gender_mapping)
            df["smoking_history"] = df["smoking_history"].map(self.smoking_mapping)
            
            for col in self.zero_invalid_cols:
                df[col] = df[col].replace(0, self.training_medians[col])

            df = df.reindex(columns=self.feature_names).astype(float)
            
            scaled = self.scaler.transform(df)
            pred = self.model.predict(scaled)[0]
            prob = float(self.model.predict_proba(scaled)[0][1])

            return {
                "prediction": "Diabetes" if pred == 1 else "No Diabetes",
                "probability": round(prob * 100, 2),
                "risk_category": "High" if prob > 0.7 else ("Medium" if prob > 0.3 else "Low"),
                "confidence": "High" if prob > 0.8 or prob < 0.2 else "Medium"
            }

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}