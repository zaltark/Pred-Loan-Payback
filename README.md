Yes, this is exactly how that text will look when rendered in Markdown.

Loan Payback Prediction (Case Study)
This machine learning project forecasts loan default risk by predicting whether a loan will be successfully paid back. The final CatBoost model achieves a 0.9302 ROC AUC, demonstrating a highly accurate and reliable method for identifying high-risk applicants.

This project showcases a complete, end-to-end data science workflow: from baseline modeling and iterative feature selection to advanced hyperparameter tuning and model evaluation.

🛠️ Technical Stack
Python: Pandas, NumPy

Modeling: CatBoost, XGBoost, LightGBM, Scikit-learn

Tuning: Optuna

Visualization: Matplotlib, Seaborn

📈 Key Results & Model Performance
The final tuned model's performance was validated against a hold-out set, proving its ability to generalize effectively.

Final Score: 0.9302 ROC AUC (achieved after 5-fold cross-validated tuning)

Predictive Power: The ROC curve demonstrates a strong ability to separate "Paid Back" (Class 1) from "Default" (Class 0).

Error Analysis: A confusion matrix was used to analyze the model's trade-offs, confirming its ability to minimize costly "False Negatives" (i.e., failing to predict a default).

Model Confidence: Prediction distribution plots show the model is highly confident in its correct predictions and reserves its "unsure" (near 0.5) predictions for the most complex, borderline cases.

🔬 My Data-Driven Workflow
This project followed a structured, iterative process to find the most optimal model and feature set.

1. Baseline Analysis & Feature Selection
First, I established a baseline by comparing four models (RandomForest, CatBoost, LightGBM, XGBoost) on the full dataset. I then used a "model-driven" approach, analyzing the combined feature importances from all models.

Key Insight: Simply selecting the top 11 most predictive features and dropping the "noise" features boosted the best model's (CatBoost) score from 0.9153 to 0.9220 AUC.

2. Iterative Engineering vs. Model Power
I hypothesized that new, manually-created features (e.g., loan_to_income) would improve performance.

Key Insight: This experiment proved that the advanced boosting models (CatBoost, XGBoost) were already identifying these complex relationships internally. The simpler, "v1" feature set remained superior. This demonstrates a key data science skill: knowing when to stop engineering and trust the model.

3. Final Hyperparameter Tuning
Using the best-performing model (CatBoost) and feature set (v1), I used Optuna to run a 50-trial optimization. This search used 5-fold cross-validation to find the most robust parameters and prevent overfitting.

This final tuning step provided the largest performance boost, increasing the model's final validation score from 0.9220 to 0.9302 AUC.

🚀 Running the Project Pipeline
This project is built as a reproducible pipeline.

Place Data: Add train.csv, test.csv, and sample_submission.csv to the data/ directory.

Run Feature Engineering: python src/01_feature_engineering.py

Run Tuning & Training: python src/02_train_and_tune.py

Run Evaluation: python src/03_evaluate_model.py
