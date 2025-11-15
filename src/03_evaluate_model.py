import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

def create_analysis_plots():
    """
    Loads the best tuned model and generates a confusion matrix, ROC curve,
    and a prediction distribution plot for analysis.
    """
    MODEL_FILE = 'models/catboost_model_tuned.joblib'
    DATA_FILE = 'data/train_refined.csv'
    OUTPUT_DIR = 'reports/figures'

    # --- Load Model and Data ---
    print("Loading tuned model and data...")
    try:
        model = joblib.load(MODEL_FILE)
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required model or data file: {e}")
        return

    # --- Prepare Data ---
    X = df.drop('loan_paid_back', axis=1)
    y = df['loan_paid_back']
    # Use the same split to get a consistent validation set
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Make Predictions ---
    print("Making predictions on validation data...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_class = model.predict(X_val)

    # --- 1. Confusion Matrix ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_val, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Paid Back', 'Paid Back'], 
                yticklabels=['Not Paid Back', 'Paid Back'])
    plt.title('Confusion Matrix for Tuned CatBoost Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = f'{OUTPUT_DIR}/confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to '{cm_path}'")
    plt.close()

    # --- 2. ROC Curve ---
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_path = f'{OUTPUT_DIR}/roc_curve.png'
    plt.savefig(roc_path)
    print(f"ROC Curve saved to '{roc_path}'")
    plt.close()

    # --- 3. Prediction Plot ---
    print("Generating Prediction Distribution Plot...")
    plot_df = pd.DataFrame({'true_label': y_val, 'predicted_probability': y_pred_proba})
    plt.figure(figsize=(10, 6))
    sns.histplot(data=plot_df, x='predicted_probability', hue='true_label', 
                 kde=True, element='step', stat='density', common_norm=False)
    plt.title('Distribution of Predicted Probabilities by True Class')
    plt.xlabel('Predicted Probability of Loan Being Paid Back')
    pred_path = f'{OUTPUT_DIR}/prediction_distribution.png'
    plt.savefig(pred_path)
    print(f"Prediction Distribution Plot saved to '{pred_path}'")
    plt.close()

if __name__ == "__main__":
    create_analysis_plots()
