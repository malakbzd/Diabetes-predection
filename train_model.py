# train_model.py - Run this once to train the model
from predictor import DiabetesPredictor

def main():
    print("🔄 Training Diabetes Prediction Model...")
    
    predictor = DiabetesPredictor()
    
    try:
        # Load and preprocess your dataset
        df = predictor.load_and_preprocess_data("diabetes.csv")
        
        # Train the model
        predictor.train_model(df)
        
        # Save the model
        predictor.save_model()
        
        print("✅ Model training completed successfully!")
        print("🎯 You can now run: python app.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()