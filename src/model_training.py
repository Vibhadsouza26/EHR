# File: src/model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_and_split_data(filepath: str, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    # Define the Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Grid search for hyperparameter tuning with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                               scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, output_file: str):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
    
    # Save metrics to a text file
    with open(output_file, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    
    # Optionally, save the trained model
    joblib.dump(model, os.path.join("..", "outputs", "best_model.joblib"))
    
if __name__ == "__main__":
    import os
    # Define file paths
    data_path = os.path.join("..", "data", "heart.csv")
    metrics_path = os.path.join("..", "outputs", "model_metrics.txt")
    
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    # Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train the model with hyperparameter tuning
    best_model = train_model(X_train_scaled, y_train)
    
    # Evaluate and save the model performance
    evaluate_model(best_model, X_test_scaled, y_test, metrics_path)
