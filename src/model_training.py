import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(filepath: str):
    """
    1. Load the data.
    2. Drop the 'id' column (if it exists).
    3. Create new features based on domain knowledge.
       - chol_age_ratio: cholesterol divided by age.
       - trestbps_age_ratio: resting blood pressure divided by age.
       - log_chol and log_oldpeak: log-transformed features (to reduce skew).
    4. Convert the target 'num' into binary: 0 if no disease (num==0), else 1.
    5. Convert all non-numeric columns into dummy variables.
    6. Impute any missing numeric values with the column mean.
    """
    df = pd.read_csv(filepath)
    print("Original Columns:", df.columns)
    
    # Drop 'id' if present
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    
    # --- Feature Engineering ---
    # Create ratio features
    df['chol_age_ratio'] = df['chol'] / df['age']
    df['trestbps_age_ratio'] = df['trestbps'] / df['age']
    # Log-transform features (adding 1 to avoid log(0))
    df['log_chol'] = np.log(df['chol'] + 1)
    df['log_oldpeak'] = np.log(df['oldpeak'] + 1)
    
    # Convert target: 0 if no heart disease, 1 otherwise
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    
    # Convert non-numeric columns to dummy variables
    df = pd.get_dummies(df, drop_first=True)
    print("Columns after get_dummies:", df.columns)
    
    # Impute missing values (if any) in numeric columns
    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Separate features and target
    X = df.drop('num', axis=1)
    y = df['num']
    return X, y

def build_pipeline():
    """
    Build a pipeline that applies:
      1. SMOTETomek resampling (to balance the classes and remove ambiguous samples),
      2. Standard scaling,
      3. XGBoost classifier.
    """
    xgb = XGBClassifier(
        eval_metric='logloss',  # to avoid warning messages
        random_state=42,
        use_label_encoder=False
    )
    pipeline = Pipeline([
        ('resample', SMOTETomek(random_state=42)),
        ('scaler', StandardScaler()),
        ('xgb', xgb)
    ])
    return pipeline

def tune_and_train_model(X_train, y_train):
    """
    Use RandomizedSearchCV to aggressively tune hyperparameters on the pipeline.
    """
    pipeline = build_pipeline()
    param_distributions = {
        'xgb__n_estimators': [100, 300, 500, 1000],
        'xgb__max_depth': [3, 5, 7, 9, 11],
        'xgb__learning_rate': [0.001, 0.01, 0.1, 0.2],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
        'xgb__min_child_weight': [1, 3, 5, 7],
        'xgb__gamma': [0, 0.1, 0.2, 0.3],
        'xgb__reg_alpha': [0, 0.01, 0.1, 1],
        'xgb__reg_lambda': [1, 1.5, 2, 5],
    }
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=100,            # try 100 random combinations
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print("Best Params:", random_search.best_params_)
    print("Best CV Accuracy:", random_search.best_score_)
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test, output_file):
    """
    Evaluate the model on the test set and save the evaluation metrics.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
    
    with open(output_file, 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "heart.csv")
    output_dir = os.path.join(script_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "model_metrics_xgb.txt")
    model_path = os.path.join(output_dir, "best_model_xgb.joblib")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(data_path)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Tune hyperparameters and train the model
    best_pipeline = tune_and_train_model(X_train, y_train)
    
    # Evaluate the final model and save metrics
    evaluate_model(best_pipeline, X_test, y_test, metrics_path)
    
    # Save the best model pipeline for future use
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model pipeline to: {model_path}")

if __name__ == "__main__":
    main()
