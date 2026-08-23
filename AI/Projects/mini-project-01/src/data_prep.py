import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple  


RANDOM_STATE = 42
TARGET_COL = "Class"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download 'creditcard.csv' from "
            f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it there."
        )
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"File '{path}' is empty.")
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame")
    
    overview = {
        "n_samples": len(df),
        "n_features": df.shape[1] - 1,  
        "columns": list(df.columns),
        "class_distribution": df[TARGET_COL].value_counts().to_dict(),
        "class_distribution_pct": (df[TARGET_COL].value_counts(normalize=True) * 100).round(4).to_dict(),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_values_per_column": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return overview


def print_overview(df: pd.DataFrame) -> None:
    
    ov = dataset_overview(df)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Number of Samples : {ov['n_samples']:,}")
    print(f"Number of Features: {ov['n_features']}")
    print(f"Missing Values    : {ov['missing_values_total']}")
    print(f"Duplicate Rows    : {ov['duplicate_rows']}")
    print("\nClass Distribution (counts):")
    for cls, cnt in ov["class_distribution"].items():
        pct = ov["class_distribution_pct"][cls]
        label = "Legitimate" if cls == 0 else "Fraudulent"
        print(f"  {cls} ({label}): {cnt:,}  ({pct:.4f}%)")
    print("\nDescriptive statistics (Amount):")
    print(df["Amount"].describe().round(4))
    print("=" * 60)


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
   
    return df.drop(columns=[TARGET_COL]).describe().round(4)


def clean_data(
    df: pd.DataFrame, 
    drop_duplicates: bool = True,
    verbose: bool = True
) -> pd.DataFrame:

    df = df.copy()
    
  
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        if verbose:
            print(f"\nFound {missing_total:,} missing values before cleaning.")
        
        for col in df.columns:
            if df[col].isnull().any():
                if col == TARGET_COL:
                  
                    before = len(df)
                    df = df.dropna(subset=[TARGET_COL])
                    if verbose:
                        print(f"  Dropped {before - len(df)} rows with missing target.")
                else:
                    
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    if verbose:
                        print(f"  Imputed '{col}' with mean: {mean_val:.4f}")
    else:
        if verbose:
            print("\nNo missing values found.")
    
   
    if drop_duplicates:
        before = len(df)
        fraud_before = df[TARGET_COL].sum()
        
        df = df.drop_duplicates().reset_index(drop=True)
        
        after = len(df)
        fraud_after = df[TARGET_COL].sum()
        
        if verbose:
            print(f"\nDropped {before - after:,} duplicate rows ({before:,} -> {after:,})")
            print(f"  Fraudulent before: {fraud_before:,}, after: {fraud_after:,}")
            print(f"  Duplicates in fraud class: {fraud_before - fraud_after:,}")
    
    return df


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    if verbose:
        print("\n" + "=" * 50)
        print("TRAIN/TEST SPLIT RESULTS")
        print("=" * 50)
        print(f"Train size: {len(X_train):,} ({len(X_train)/len(y)*100:.1f}%)")
        print(f"Test size : {len(X_test):,} ({len(X_test)/len(y)*100:.1f}%)")
        print("\nTrain class distribution:")
        print(f"  Legitimate (0): {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.4f}%)")
        print(f"  Fraudulent (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.4f}%)")
        print("\nTest class distribution:")
        print(f"  Legitimate (0): {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.4f}%)")
        print(f"  Fraudulent (1): {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.4f}%)")
        print("=" * 50)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
    df = load_data(DATA_PATH)
    print_overview(df)