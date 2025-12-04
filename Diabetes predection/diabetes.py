import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pickle
import json
import os
from datetime import datetime
import logging


# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiabetesPredictor:
    # =========================================================
    # Initialization
    # =========================================================
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_medians = None
        self.is_trained = False

        # Allowed categorical values
        self.valid_genders = ["female", "male", "other"]
        self.valid_smoking_values = [
            "never", "not current", "current", "no info", "ever", "former"
        ]

        # Mappings
        self.gender_mapping = {'female': 0, 'male': 1, 'other': 2}
        self.smoking_mapping = {
            'never': 0, 'not current': 1, 'current': 2,
            'no info': 3, 'ever': 4, 'former': 5
        }

        self.zero_invalid_cols = ['bmi', 'hbA1c_level', 'blood_glucose_level']

    # =========================================================
    # 1️⃣ Dataset Loading & Cleaning
    # =========================================================
    def load_and_preprocess_data(self, file_path="diabetes_dataset_with_notes.csv"):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded with {len(df)} rows.")

            drop_cols = ["year", "location", "clinical_notes"]
            df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

            df = self._clean_data(df)

            if "diabetes" not in df.columns:
                raise ValueError("Missing target column: 'diabetes'")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def _clean_data(self, df):
        """Clean training data while recording medians."""
        self.training_medians = {}

        # Replace zeros with median
        for col in self.zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                median_val = df[col].median()

                if pd.isna(median_val):
                    raise ValueError(f"Column '{col}' contains no non-zero values, cannot compute median.")

                df[col] = df[col].fillna(median_val)
                self.training_medians[col] = float(median_val)

        # Normalize category strings
        df["gender"] = (
            df["gender"].astype(str)
            .str.strip().str.lower()
            .map(self.gender_mapping)
            .fillna(self.gender_mapping["other"])
        )

        df["smoking_history"] = (
            df["smoking_history"].astype(str)
            .str.strip().str.lower()
            .map(self.smoking_mapping)
            .fillna(self.smoking_mapping["no info"])
        )

        return df

    # =========================================================
    # 2️⃣ Train Model
    # =========================================================
    def train_model(self, df, test_size=0.2, random_state=42):
        try:
            X = df.drop("diabetes", axis=1)
            y = df["diabetes"]
            self.feature_names = list(X.columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Scale numerical features
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

            self._evaluate_model(X_test_scaled, y_test)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    # =========================================================
    # Evaluation
    # =========================================================
    def _evaluate_model(self, X_test_scaled, y_test):
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        print("\n========== MODEL EVALUATION ==========")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # =========================================================
    # 3️⃣ Save Model + Metadata
    # =========================================================
    def save_model(self):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        pickle.dump(self.model, open("diabetes_model.pkl", "wb"))
        pickle.dump(self.scaler, open("scaler.pkl", "wb"))

        metadata = {
            "feature_names": self.feature_names,
            "training_medians": self.training_medians,
            "timestamp": datetime.now().isoformat()
        }

        json.dump(metadata, open("model_metadata.json", "w"), indent=2)
        logger.info("Model, scaler, and metadata saved successfully.")

    # =========================================================
    # 4️⃣ Load Model
    # =========================================================
    def load_model(self):
        try:
            if not os.path.exists("diabetes_model.pkl"):
                return False

            self.model = pickle.load(open("diabetes_model.pkl", "rb"))
            self.scaler = pickle.load(open("scaler.pkl", "rb"))

            if os.path.exists("model_metadata.json"):
                meta = json.load(open("model_metadata.json"))

                self.feature_names = meta.get("feature_names")
                self.training_medians = meta.get("training_medians", {})

                if not self.feature_names:
                    raise ValueError("feature_names missing in metadata")

            else:
                raise FileNotFoundError("metadata.json missing")

            self.is_trained = True
            logger.info("Model loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    # =========================================================
    # 5️⃣ **Robust Prediction with Full Validation**
    # =========================================================
    def predict_diabetes(self, user_input: dict) -> dict:
        if not self.is_trained:
            return {"error": "Model not trained or loaded."}

        # ---------- VALIDATION ----------
        try:
            # Required fields
            required_fields = [
                "gender", "age", "bmi", "smoking_history",
                "hbA1c_level", "blood_glucose_level"
            ]

            missing = [f for f in required_fields if f not in user_input]
            if missing:
                return {"error": f"Missing fields: {', '.join(missing)}"}

            # Validate gender
            gender = str(user_input["gender"]).strip().lower()
            if gender not in self.valid_genders:
                return {
                    "error": f"Invalid gender '{user_input['gender']}'. "
                             f"Allowed: {self.valid_genders}"
                }

            # Validate smoking history
            smoking = str(user_input["smoking_history"]).strip().lower()
            if smoking not in self.valid_smoking_values:
                return {
                    "error": f"Invalid smoking history '{user_input['smoking_history']}'. "
                             f"Allowed: {self.valid_smoking_values}"
                }

            # Validate numeric fields
            numeric_fields = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]
            validated = {}
            for field in numeric_fields:
                try:
                    validated[field] = float(user_input[field])
                except Exception:
                    return {"error": f"Field '{field}' must be a number."}

            # ---------- BUILD CLEAN DATAFRAME ----------
            df = pd.DataFrame([{
                "gender": self.gender_mapping[gender],
                "age": validated["age"],
                "bmi": validated["bmi"] if validated["bmi"] != 0 else self.training_medians.get("bmi"),
                "smoking_history": self.smoking_mapping[smoking],
                "hbA1c_level": validated["hbA1c_level"] if validated["hbA1c_level"] != 0 else self.training_medians.get("hbA1c_level"),
                "blood_glucose_level": validated["blood_glucose_level"] if validated["blood_glucose_level"] != 0 else self.training_medians.get("blood_glucose_level"),
            }])

            # Order columns
            df = df.reindex(columns=self.feature_names)

        except Exception as e:
            return {"error": f"Input preparation failed: {e}"}

        # ---------- PREDICT ----------
        try:
            scaled = self.scaler.transform(df)
            pred = int(self.model.predict(scaled)[0])
            prob = float(self.model.predict_proba(scaled)[0][1])

            return {
                "prediction": "Diabetes" if pred == 1 else "No Diabetes",
                "probability": round(prob * 100, 2),
                "confidence": "High" if prob > 0.8 or prob < 0.2 else "Medium",
                "risk_category": "High" if prob > 0.7 else ("Medium" if prob > 0.3 else "Low")
            }

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}


# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    predictor = DiabetesPredictor()

    # Try loading existing model; else train new one
    if not predictor.load_model():
        print("Training new model...")
        df = predictor.load_and_preprocess_data()
        predictor.train_model(df)
        predictor.save_model()

    print("\nWelcome to the Diabetes Predictor!")
    print("Please enter the following information:")

    try:
        user_input = {
            "gender": input("Gender (Female/Male/Other): ").strip(),
            "age": input("Age (0-120): ").strip(),
            "bmi": input("BMI (10-60): ").strip(),
            "smoking_history": input(
                "Smoking History (never/not current/current/no info/ever/former): "
            ).strip(),
            "hbA1c_level": input("HbA1c Level (3-15): ").strip(),
            "blood_glucose_level": input("Blood Glucose Level (50-300): ").strip(),
        }
    except Exception:
        print("Input could not be read. Please try again.")
        return

    response = predictor.predict_diabetes(user_input)

    print("\n======= RESULT ========")
    print(response)


if __name__ == "__main__":
    main()
