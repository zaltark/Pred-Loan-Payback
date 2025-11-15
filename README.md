# Loan Payback Prediction Project

This project aims to predict whether a loan will be paid back based on a set of features. The project follows a typical data science workflow, including exploratory data analysis, feature engineering, model training, and hyperparameter tuning.

## Project Structure

```
/
|-- data/
|   |-- train.csv                 # Original training data
|   |-- test.csv                  # Original test data
|   |-- sample_submission.csv     # Sample submission file
|   |-- train_refined.csv         # Refined training data after feature selection
|   `-- test_refined.csv          # Refined test data after feature selection
|-- src/
|   |-- 01_feature_engineering.py # Script to generate the refined datasets
|   |-- 02_train_and_tune.py      # Script to tune the CatBoost model and train the final model
|   `-- 03_evaluate_model.py      # Script to generate analysis plots for the final model
|-- exploration/
|   `-- ...                     # Scripts used for iterative development and exploration
|-- models/
|   `-- catboost_tuned.joblib     # The final tuned CatBoost model
|-- submissions/
|   `-- submission_catboost_tuned.csv # The final submission file
|-- reports/
|   `-- figures/
|       |-- confusion_matrix.png
|       |-- roc_curve.png
|       |-- prediction_distribution.png
|       `-- correlation_heatmap.png
`-- README.md
```

## Project Journey & Key Learnings

This project followed an iterative process of feature engineering and model tuning to arrive at the final solution.

1.  **Initial Baseline:** We began by establishing baseline performance with four different models: RandomForest, CatBoost, LightGBM, and XGBoost on the fully one-hot encoded dataset. The boosting models, particularly XGBoost, showed the best initial performance.

2.  **Feature Selection (v1):** We analyzed the combined feature importances from all baseline models and selected the top 11 most influential features. Retraining the models on this "refined" dataset led to a significant performance increase across the board. The best model in this stage was **CatBoost with a ROC AUC of 0.9220**.

3.  **Advanced Feature Engineering (v2 & v3):** We experimented with creating new interaction and ratio features (e.g., `loan_to_income`, `dti_credit_interaction`). While these new features showed high importance, they did not lead to an overall improvement in the top models' performance compared to the v1 feature set. This was a key learning: more complex features are not always better, and the initial feature selection was highly effective.

4.  **Hyperparameter Tuning:** We took the best performing model and feature set (CatBoost on the v1 refined data) and used Optuna for hyperparameter tuning. This final step provided the largest performance boost, resulting in a final model with a **ROC AUC of 0.9302** on the validation set.

This journey highlights the importance of a structured, iterative approach, from establishing a strong baseline to methodical feature selection and finally, fine-tuning the best performing model.

## Data

The training and test data (`train.csv`, `test.csv`, `sample_submission.csv`) are not included in this repository due to their size. Please place them in the `data/` directory before running the scripts.

## Workflow

1.  **Feature Engineering (`src/01_feature_engineering.py`):**
    *   This script loads the original data from the `data/` directory.
    *   It performs feature selection based on a combined feature importance from multiple models.
    *   It creates the `train_refined.csv` and `test_refined.csv` datasets and saves them in the `data/` directory.

2.  **Model Training and Tuning (`src/02_train_and_tune.py`):**
    *   This script uses the `optuna` library to perform hyperparameter tuning on a CatBoost model using the refined data.
    *   It uses 5-fold cross-validation to find the best hyperparameters for AUC.
    *   It then trains a final CatBoost model on the full refined dataset with the best hyperparameters.
    *   The final model is saved to `models/catboost_tuned.joblib`.
    *   A submission file is generated at `submissions/submission_catboost_tuned.csv`.

3.  **Model Evaluation (`src/03_evaluate_model.py`):**
    *   This script loads the final tuned model and generates a set of analysis plots:
        *   Confusion Matrix
        *   ROC Curve
        *   Prediction Distribution Plot
    *   The plots are saved in `reports/figures/`.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn catboost xgboost lightgbm optuna seaborn matplotlib
    ```

2.  **Place data:**
    *   Place `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/` directory.

3.  **Run the pipeline:**
    ```bash
    python src/01_feature_engineering.py
    python src/02_train_and_tune.py
    python src/03_evaluate_model.py
    ```

## Results

The final tuned CatBoost model achieved a ROC AUC score of **0.9302** on the validation set. The analysis plots in `reports/figures/` provide a detailed look at the model's performance.
