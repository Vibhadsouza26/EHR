import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset into a DataFrame and drop the 'id' column if present.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
            print("Dropped 'id' column as it does not add analytical value.")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {filepath}: {e}")

def basic_info(df: pd.DataFrame):
    """
    Print basic information: shape, data types, duplicate count, and missing values.
    """
    print("\n--- BASIC DATA INFO ---")
    print(f"Shape of DataFrame: {df.shape}")
    print("Data Types:\n", df.dtypes)
    dup_count = df.duplicated().sum()
    print(f"Number of Duplicates: {dup_count}")
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by imputing missing numeric values with the mean.
    If any rows still contain missing values (e.g. in non-numeric columns),
    drop those rows.
    """
    print("\n--- PREPROCESSING ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"Imputed missing values in '{col}' with mean: {mean_val:.2f}")
    null_rows_before = df.isnull().any(axis=1).sum()
    if null_rows_before > 0:
        print(f"Dropping {null_rows_before} rows still containing missing values.")
        df.dropna(inplace=True)
    else:
        print("No rows needed to be dropped after imputation.")
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all remaining non-numeric columns (object, category, bool) into numeric
    using get_dummies so that all features can be included in correlation analysis.
    """
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(cat_cols) > 0:
        print("\n--- ENCODING CATEGORICAL FEATURES ---")
        print("Categorical Columns:", list(cat_cols))
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        print("No categorical columns to encode.")
    return df

def explore_numerical(df: pd.DataFrame, output_dir: str):
    """
    Create histograms, box plots, and violin plots for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Histogram with KDE
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, color='blue')
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        hist_path = os.path.join(output_dir, f"{col}_hist.png")
        plt.savefig(hist_path)
        plt.close()
        
        # Box plot
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color='green')
        plt.title(f"Box Plot of {col}")
        plt.tight_layout()
        box_path = os.path.join(output_dir, f"{col}_box.png")
        plt.savefig(box_path)
        plt.close()
        
        # Violin plot
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=df[col], color='purple')
        plt.title(f"Violin Plot of {col}")
        plt.tight_layout()
        violin_path = os.path.join(output_dir, f"{col}_violin.png")
        plt.savefig(violin_path)
        plt.close()

def explore_categorical(df: pd.DataFrame, output_dir: str):
    """
    Create count plots for any columns that remain categorical.
    (After encoding, this might be empty.)
    """
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[col], palette='Set2')
        plt.title(f"Count Plot of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        cat_path = os.path.join(output_dir, f"{col}_countplot.png")
        plt.savefig(cat_path)
        plt.close()

def correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Generate a correlation heatmap of all numeric columns (including those created
    by dummy encoding). The figure size adjusts dynamically based on the number of columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns for a correlation heatmap (need at least 2).")
        return
    corr = numeric_df.corr()
    fig_width = max(10, len(numeric_df.columns) * 0.5)
    fig_height = max(8, len(numeric_df.columns) * 0.5)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_path}")

def advanced_eda(df: pd.DataFrame, output_dir: str):
    """
    Create a pairplot of numeric variables if there are 10 or fewer columns.
    This visualizes pairwise relationships and distributions.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] <= 10:
        sns.pairplot(numeric_df, diag_kind='kde')
        pairplot_path = os.path.join(output_dir, "pairplot.png")
        plt.savefig(pairplot_path)
        plt.close()
        print(f"Pairplot saved to {pairplot_path}")
    else:
        print("Skipping pairplot (too many numeric columns).")

def perform_eda(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run all EDA steps and save the resulting plots into the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe(include='all'))
    explore_numerical(df, output_dir)
    explore_categorical(df, output_dir)
    correlation_heatmap(df, output_dir)
    advanced_eda(df, output_dir)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "heart.csv")
    figures_dir = os.path.join(script_dir, "..", "outputs", "figures")
    
    df = load_data(data_path)
    basic_info(df)
    df = preprocess_data(df)
    df = encode_categorical(df)
    perform_eda(df, figures_dir)

if __name__ == "__main__":
    main()
