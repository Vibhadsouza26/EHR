import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset into a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {filepath}: {e}")

def perform_eda(df: pd.DataFrame, output_dir: str) -> None:
    """Generate and save plots for EDA."""
    os.makedirs(output_dir, exist_ok=True)
    print("Summary Statistics:")
    print(df.describe())
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_path}")

def preprocess_data(df: pd.DataFrame):
    """Basic preprocessing: check for missing values."""
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Proceeding with imputation or removal.")
        df = df.dropna()  # For demonstration; consider better strategies.
    else:
        print("No missing values found.")
    return df

if __name__ == "__main__":
    # Determine the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to the script directory
    data_path = os.path.join(script_dir, "..", "data", "heart.csv")
    figures_dir = os.path.join(script_dir, "..", "outputs", "figures")
    
    # Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)
    perform_eda(df, figures_dir)
