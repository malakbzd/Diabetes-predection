import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
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


# =========================================================
# 🎯 Diabetes Predictor Class
# =========================================================
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

    # -----------------------------------------------------
    # 1️⃣ Load + Clean Data
    # -----------------------------------------------------
    def load_and_preprocess_data(self, file_path="diabetes_dataset_with_notes.csv"):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded with {len(df)} rows.")

            # Drop unused columns if they exist
            drop_cols = ["year", "location", "clinical_notes"]
            df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

            df = self._clean_data(df)

            if "diabetes" not in df.columns:
                raise ValueError("Dataset missing target column 'diabetes'")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def _clean_data(self, df):
        self.training_medians = {}

        for col in self.zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

                self.training_medians[col] = float(median_val)

        # Encode categorical values
        df["gender"] = df["gender"].map(self.gender_mapping).fillna(self.gender_mapping["Other"])
        df["smoking_history"] = df["smoking_history"].map(self.smoking_mapping).fillna(
            self.smoking_mapping["No Info"]
        )

        return df

    # -----------------------------------------------------
    # 2️⃣ Train Model
    # -----------------------------------------------------
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

            # Hyperparameter tuning
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

            self._evaluate_model(X, y, X_test_scaled, y_test)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    # -----------------------------------------------------
    # Model evaluation
    # -----------------------------------------------------
    def _evaluate_model(self, X, y, X_test_scaled, y_test):
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        print("\n========== MODEL EVALUATION ==========")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        fi = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": self.model.feature_importances_
        }).sort_values("Importance", ascending=False)

        print("\n========== FEATURE IMPORTANCE ==========")
        print(fi)

    # -----------------------------------------------------
    # 3️⃣ Save Model + Metadata
    # -----------------------------------------------------
    def save_model(self):
        if not self.is_trained:
            raise ValueError("You must train the model before saving.")

        pickle.dump(self.model, open("diabetes_model.pkl", "wb"))
        pickle.dump(self.scaler, open("scaler.pkl", "wb"))

        metadata = {
            "feature_names": self.feature_names,
            "training_medians": self.training_medians,
            "timestamp": datetime.now().isoformat()
        }

        json.dump(metadata, open("model_metadata.json", "w"), indent=2)
        logger.info("Model + scaler + metadata saved.")

    # -----------------------------------------------------
    # 4️⃣ Load Model (with fixed feature name recovery)
    # -----------------------------------------------------
    def load_model(self):
        if not (os.path.exists("diabetes_model.pkl") and os.path.exists("scaler.pkl")):
            logger.warning("Model files not found.")
            return False

        self.model = pickle.load(open("diabetes_model.pkl", "rb"))
        self.scaler = pickle.load(open("scaler.pkl", "rb"))

        # Try metadata first
        if os.path.exists("model_metadata.json"):
            try:
                meta = json.load(open("model_metadata.json"))

                if "feature_names" in meta and meta["feature_names"]:
                    self.feature_names = meta["feature_names"]
                else:
                    raise Exception("feature_names missing")

                self.training_medians = meta.get("training_medians", {})
                logger.info("Metadata loaded successfully.")

            except Exception as e:
                logger.warning(f"Metadata invalid ({e}). Recovering feature names.")
                self._recover_feature_names()

        else:
            logger.warning("metadata.json not found — reconstructing.")
            self._recover_feature_names()

        self.is_trained = True
        logger.info("Model loaded successfully.")
        return True

    def _recover_feature_names(self):
        """Fallback when metadata is missing/corrupted"""
        if hasattr(self.scaler, "mean_"):
            n = len(self.scaler.mean_)
            self.feature_names = [f"feature_{i}" for i in range(n)]
            logger.info("Feature names reconstructed from scaler.")
        else:
            raise ValueError("Cannot recover feature names.")

    # -----------------------------------------------------
    # 5️⃣ Prediction
    # -----------------------------------------------------
    def predict_diabetes(self, user_input: dict) -> dict:
        if not self.is_trained:
            return {"error": "Model not trained or loaded."}

        # Convert input into a row
        try:
            df = pd.DataFrame([user_input])

            df["gender"] = df["gender"].map(self.gender_mapping)
            df["smoking_history"] = df["smoking_history"].map(self.smoking_mapping)

            for col in self.zero_invalid_cols:
                df[col] = df[col].replace(0, self.training_medians[col])

            df = df.reindex(columns=self.feature_names).astype(float)

        except Exception as e:
            return {"error": f"Input validation failed: {e}"}

        try:
            scaled = self.scaler.transform(df)
            pred = self.model.predict(scaled)[0]
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

    # Load existing model or train a new one
    if not predictor.load_model():
        df = predictor.load_and_preprocess_data()
        predictor.train_model(df)
        predictor.save_model()

    print("\nWelcome to the Diabetes Predictor!")
    print("Please enter the following information:")

    # Collect input interactively
    user_input = {}
    try:
        user_input['gender'] = input("Gender (Female/Male): ").strip()
        user_input['age'] = float(input("Age (0-120): ").strip())
        user_input['bmi'] = float(input("BMI (10-60): ").strip())
        user_input['smoking_history'] = input(
            "Smoking History (never/not current/current/No Info/ever/former): "
        ).strip()
        user_input['hbA1c_level'] = float(input("HbA1c Level (3-15): ").strip())
        user_input['blood_glucose_level'] = float(input("Blood Glucose Level (50-300): ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values where required.")
        return

    # Make prediction
    result = predictor.predict_diabetes(user_input)

    print("\nPrediction Result:")
    print(result)


if __name__ == "__main__":
    main()
