import pandas as pd
import joblib
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# --- Global Variables ---
TRAIN_FILE = 'data/train_refined.csv'
TEST_FILE = 'data/test_refined.csv'
MODEL_FILE = 'models/catboost_model_tuned.joblib'
SUBMISSION_FILE = 'submissions/submission_catboost_tuned.csv'
N_SPLITS = 5
N_TRIALS = 50

def objective(trial):
    """
    The objective function for Optuna optimization.
    """
    # Load the training data
    try:
        df = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print(f"Error: '{TRAIN_FILE}' not found.")
        # Optuna will see this as a failed trial
        raise optuna.exceptions.TrialPruned()

    X = df.drop('loan_paid_back', axis=1)
    y = df['loan_paid_back']

    # --- Hyperparameter Search Space ---
    params = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0,
        'iterations': 1000, # High number for early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=0
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        auc_scores.append(auc)

    return np.mean(auc_scores)

def tune_and_train():
    """
    Runs the Optuna study, finds the best hyperparameters, and trains a final model.
    """
    print(f"Starting Optuna hyperparameter tuning for CatBoost ({N_TRIALS} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n--- Tuning Complete ---")
    print(f"Best trial AUC: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # --- Train Final Model with Best Hyperparameters ---
    print("\nTraining final CatBoost model with best hyperparameters...")
    best_params = study.best_params
    best_params.update({
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0,
        'iterations': 2000 # Higher iterations for the final model
    })

    df = pd.read_csv(TRAIN_FILE)
    X = df.drop('loan_paid_back', axis=1)
    y = df['loan_paid_back']
    
    final_model = CatBoostClassifier(**best_params)
    # No early stopping for the final model, train on all data
    final_model.fit(X, y)

    # Save the tuned model
    joblib.dump(final_model, MODEL_FILE)
    print(f"Tuned model saved as '{MODEL_FILE}'.")

    # --- Generate Submission File ---
    print("\nGenerating submission file with tuned model...")
    test_df = pd.read_csv(TEST_FILE)
    test_ids = test_df['id']
    X_test = test_df.drop('id', axis=1)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    submission_df = pd.DataFrame({'id': test_ids, 'loan_paid_back': test_probabilities})
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file '{SUBMISSION_FILE}' created successfully.")

if __name__ == "__main__":
    tune_and_train()
