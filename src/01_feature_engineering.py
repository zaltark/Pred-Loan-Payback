import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

def feature_engineering():
    """
    Loads the original data, performs one-hot encoding, selects the best
    subset of features (from the v1 analysis), and saves the refined datasets.
    """
    print("Starting feature engineering...")

    # --- Load Original Data ---
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
    except FileNotFoundError:
        print("Error: Original data not found in 'data/' directory.")
        return

    # --- Feature Selection ---
    # These were the top features from the first round of analysis that led to the best model
    features_to_keep_one_hot = [
        'debt_to_income_ratio',
        'employment_status_Unemployed',
        'credit_score',
        'loan_amount',
        'annual_income',
        'interest_rate',
        'employment_status_Employed',
        'employment_status_Student',
        'employment_status_Retired',
        'employment_status_Self-employed',
        'loan_purpose_Debt consolidation'
    ]
    
    # Identify original columns needed for these features
    # This requires knowing which original columns create the one-hot encoded ones
    numerical_features_needed = [
        'debt_to_income_ratio', 'credit_score', 'loan_amount', 
        'annual_income', 'interest_rate'
    ]
    categorical_features_needed = ['employment_status', 'loan_purpose']

    # --- Preprocessing ---
    # We need to fit an encoder on the training data and transform both train and test
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df[categorical_features_needed])
    
    # Transform training data
    encoded_features_train = encoder.transform(train_df[categorical_features_needed])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features_needed)
    encoded_df_train = pd.DataFrame(encoded_features_train, columns=encoded_feature_names, index=train_df.index)
    
    # Combine with numerical features for the full processed training set
    X_train_processed = pd.concat([train_df[numerical_features_needed], encoded_df_train], axis=1)
    
    # Ensure all desired columns exist, fill with 0 if not
    for col in features_to_keep_one_hot:
        if col not in X_train_processed.columns:
            X_train_processed[col] = 0
            
    # Select the final subset of features
    X_train_refined = X_train_processed[features_to_keep_one_hot]
    
    # Add the target variable back
    train_refined_df = pd.concat([X_train_refined, train_df['loan_paid_back']], axis=1)
    
    # Save the refined training data
    train_refined_df.to_csv('data/train_refined.csv', index=False)
    print("Refined training dataset 'data/train_refined.csv' created successfully.")

    # Transform test data
    test_ids = test_df['id']
    encoded_features_test = encoder.transform(test_df[categorical_features_needed])
    encoded_df_test = pd.DataFrame(encoded_features_test, columns=encoded_feature_names, index=test_df.index)
    
    X_test_processed = pd.concat([test_df[numerical_features_needed], encoded_df_test], axis=1)
    
    for col in features_to_keep_one_hot:
        if col not in X_test_processed.columns:
            X_test_processed[col] = 0
            
    X_test_refined = X_test_processed[features_to_keep_one_hot]
    
    test_refined_df = pd.concat([test_ids, X_test_refined], axis=1)
    
    test_refined_df.to_csv('data/test_refined.csv', index=False)
    print("Refined testing dataset 'data/test_refined.csv' created successfully.")

if __name__ == "__main__":
    feature_engineering()