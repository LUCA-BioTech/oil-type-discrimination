"""
Prediction script for oil type discrimination model.

Loads a trained model and makes predictions on new data.
"""

import joblib
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import unified utilities
from src.utils import load_data_from_csv
from src.config import FEATURE_COLUMNS


def predict(model_path: str, input_csv: str, output_path: str = "results/predictions/pred_result.csv"):
    """
    Load a trained model and make predictions on new data.

    Args:
        model_path: Path to the saved model file (.pkl or .joblib)
        input_csv: Path to the CSV file with data to predict
        output_path: Path to save prediction results (default: results/predictions/pred_result.csv)
    """
    print("="*60)
    print("PREDICTION")
    print("="*60)

    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)

    # Load data using unified utility
    print(f"Loading data from: {input_csv}")
    X, _ = load_data_from_csv(input_csv)
    print(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")

    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X)

    # Load original CSV to get all columns
    import pandas as pd
    df = pd.read_csv(input_csv)

    # Add predictions to dataframe
    # Decode predictions if they are encoded
    if hasattr(model, 'get_classes'):
        class_names = model.get_classes()
        df['predicted_label'] = [class_names[p] for p in predictions]
    else:
        df['predicted_label'] = predictions

    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    if hasattr(model, 'get_classes'):
        for class_name in model.get_classes():
            count = (df['predicted_label'] == class_name).sum()
            print(f"  {class_name}: {count} samples")
    else:
        print(df['predicted_label'].value_counts().to_string())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.predict <model_path> <input_csv> [output_csv]")
        print("\nExample:")
        print("  python -m src.predict models/model.pkl data/new_data.csv")
        print("  python -m src.predict models/model.pkl data/new_data.csv results/predictions/my_predictions.csv")
        sys.exit(1)

    model_path = sys.argv[1]
    input_csv = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "results/predictions/pred_result.csv"

    predict(model_path, input_csv, output_path)
