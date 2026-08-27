import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List
import logging


RANDOM_STATE = 42
TARGET_COL = "Class"


def load_data(path: str, **read_csv_kwargs) -> pd.DataFrame:

    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download 'creditcard.csv' from "
            f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it there."
        )
    
    try:
        
        kwargs = {'low_memory': False}
        kwargs.update(read_csv_kwargs)
        df = pd.read_csv(file_path, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{path}': {e}") from e
    
    if df.empty:
        raise ValueError(f"File '{path}' is empty or contains no data.")
    
    logging.info(f"Successfully loaded dataset with {len(df):,} rows and {len(df.columns)} columns.")
    return df


def dataset_overview(df: pd.DataFrame, target_col: str = None) -> dict:

    if target_col is None:
        target_col = TARGET_COL
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    overview = {
        "n_samples": len(df),
        "n_features": df.shape[1] - 1,
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "class_distribution": df[target_col].value_counts().to_dict(),
        "class_distribution_pct": (df[target_col].value_counts(normalize=True) * 100).round(4).to_dict(),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_values_per_column": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    

    overview["target_column"] = target_col
    
    logging.info(f"Dataset overview generated: {len(df):,} samples, {overview['n_features']} features")
    return overview  


def get_descriptive_stats(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
   
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if not numerical_cols:
        raise ValueError("No numerical columns found for descriptive statistics.")
    
    stats_df = df[numerical_cols].describe().round(4)
    logging.info(f"Descriptive statistics generated for {len(numerical_cols)} numerical features.")
    return stats_df


def clean_data(
    df: pd.DataFrame, 
    drop_duplicates: bool = True,
    handle_missing: str = 'drop', 
    verbose: bool = True
) -> pd.DataFrame:

    df = df.copy()
    initial_rows = len(df)
    
    
    if drop_duplicates:
        df = df.drop_duplicates()
        if verbose:
            removed = initial_rows - len(df)
            logging.info(f"Removed {removed:,} duplicate rows.")
    
    
    target_col = TARGET_COL  
    missing_target = df[target_col].isnull().sum()
    if missing_target > 0:
        before = len(df)
        df = df.dropna(subset=[target_col])
        if verbose:
            logging.info(f"Dropped {before - len(df)} rows with missing target values.")
    

    missing_cols = [col for col in df.columns if col != target_col and df[col].isnull().any()]
    
    if missing_cols:
        if handle_missing == 'drop':
            before = len(df)
            df = df.dropna(subset=missing_cols)
            if verbose:
                logging.info(f"Dropped {before - len(df)} rows with missing values in columns: {missing_cols}")
        
        elif handle_missing == 'fill':
            for col in missing_cols:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:

                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            if verbose:
                logging.info(f"Filled missing values in columns: {missing_cols}")

        else:
            logging.warning(f"Unknown handle_missing strategy: '{handle_missing}'. Skipping missing value handling.")        
    else:
        if verbose:
            logging.info("No missing values found in feature columns.")
    
    if verbose:
        logging.info(f"Data cleaning complete. {len(df):,} rows remaining (from {initial_rows:,}).")
    
    return df


def split_features_target(
    df: pd.DataFrame, 
    target_col: str = TARGET_COL,
    sample_size: int = None,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    
    if sample_size is not None:
        if sample_size > len(X):
            raise ValueError(
                f"sample_size ({sample_size}) cannot be larger than dataset size ({len(X)})"
            )
        
        if sample_size < len(X):
            X, _, y, _ = train_test_split(
                X, y,
                train_size=sample_size,
                stratify=y,
                random_state=random_state
            )
            logging.info(f"Sampled {sample_size:,} rows while preserving class distribution.")
    
    logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must not be empty.")
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}")
    if len(y.unique()) < 2:
        raise ValueError(f"Need at least 2 classes for stratification. Found: {y.unique()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state,
        shuffle=shuffle
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

def fit_scaler(
    X_train: pd.DataFrame, 
    columns: Optional[List[str]] = None
) -> StandardScaler:
    if columns is None:
        columns = X_train.columns.tolist()
    else:
        
        missing = [col for col in columns if col not in X_train.columns]
        if missing:
            raise ValueError(f"Columns not found in X_train: {missing}")
    
    scaler = StandardScaler()
    scaler.fit(X_train[columns])
    scaler.feature_names_ = list(columns)
    
    logging.info(f"Scaler fitted on {len(columns)} columns: {columns}")
    return scaler

def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:

   
    if not hasattr(scaler, 'feature_names_'):
        raise ValueError(
            "Scaler has not been fitted yet. Call fit_scaler() first."
        )
    
    X = X.copy()
    cols = scaler.feature_names_
    

    missing_cols = [col for col in cols if col not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")
    
  
    X[cols] = scaler.transform(X[cols])
    
    logging.info(f"Scaler applied to {len(cols)} columns.")
    return X


def prepare_pipeline_data(
    csv_path: str, 
    test_size: float = 0.2, 
    random_state: int = RANDOM_STATE,
    scale_columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> Dict:

    if verbose:
        print("Step 1/5: Loading data...")
    df = load_data(csv_path)
    
    if verbose:
        print("Step 2/5: Cleaning data...")
    df = clean_data(df, verbose=verbose)
    
    if verbose:
        print("Step 3/5: Splitting features and target...")
    X, y = split_features_target(
        df, 
        sample_size=sample_size,
        random_state=random_state
    )
    
    if verbose:
        print("Step 4/5: Stratified train/test split...")
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size, random_state, verbose=verbose
    )
    
    if verbose:
        print("Step 5/5: Fitting scaler on training data and transforming...")
    scaler = fit_scaler(X_train, columns=scale_columns)
    X_train_scaled = apply_scaler(scaler, X_train)
    X_test_scaled = apply_scaler(scaler, X_test)
    
    if verbose:
        print("Data preparation complete!\n")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


def overview_to_markdown(overview: dict) -> str:

    lines = []
    
    # Header
    lines.append("## 📊 Dataset Overview\n")
    
    # Basic statistics
    lines.append("### General Statistics\n")
    lines.append(f"- **Number of Samples:** `{overview['n_samples']:,}`")
    lines.append(f"- **Number of Features:** `{overview['n_features']}`")
    lines.append(f"- **Number of Columns:** `{overview['n_columns']}`")
    lines.append(f"- **Target Column:** `{overview['target_column']}`")
    lines.append(f"- **Duplicate Rows:** `{overview['duplicate_rows']:,}`")
    lines.append(f"- **Total Missing Values:** `{overview['missing_values_total']:,}`\n")
    
    # Class distribution table
    lines.append("### Class Distribution\n")
    lines.append("| Class | Count | Percentage |")
    lines.append("|-------|------:|-----------:|")
    
    for cls, count in overview['class_distribution'].items():
        pct = overview['class_distribution_pct'][cls]
        # Add emoji for fraud detection
        label = "Fraud" if cls == 1 else "Legitimate"
        lines.append(f"| {label} | `{count:,}` | `{pct:.2f}%` |")
    
    # Imbalance warning
    fraud_pct = overview['class_distribution_pct'].get(1, 0)
    if fraud_pct < 5:
        lines.append("\n> **Note:** This dataset is **highly imbalanced**. Accuracy alone is not a reliable metric.")
    
    # Missing values (if any)
    missing_per_col = {k: v for k, v in overview['missing_values_per_column'].items() if v > 0}
    if missing_per_col:
        lines.append("\n### Missing Values by Column\n")
        lines.append("| Column | Missing Count |")
        lines.append("|--------|--------------:|")
        for col, count in missing_per_col.items():
            lines.append(f"| `{col}` | `{count:,}` |")
    else:
        lines.append("\n**No missing values found.**")
    
    return "\n".join(lines)


def descriptive_stats_to_markdown(stats_df: pd.DataFrame) -> str:

    lines = []
    lines.append("### Descriptive Statistics (Numerical Features)\n")
    lines.append("| Statistic | " + " | ".join(stats_df.columns) + " |")
    lines.append("|-----------|" + "|".join(["------:" for _ in stats_df.columns]) + "|")
    
    for index, row in stats_df.iterrows():
        row_values = [f"`{val:.4f}`" if isinstance(val, (int, float)) else str(val) for val in row.values]
        lines.append(f"| {index} | " + " | ".join(row_values) + " |")
    
    return "\n".join(lines)


def generate_full_report_markdown(df: pd.DataFrame, target_col: str = None) -> str:

    if target_col is None:
        target_col = TARGET_COL
    

    overview = dataset_overview(df, target_col=target_col)
    overview_md = overview_to_markdown(overview)
    
  
    stats_df = get_descriptive_stats(df, target_col=target_col)
    stats_md = descriptive_stats_to_markdown(stats_df)
    

    full_report = overview_md + "\n\n" + stats_md
    return full_report


def save_report_to_file(df: pd.DataFrame, output_path: str = "README.md") -> None:

    report = generate_full_report_markdown(df)
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Report saved to {output_path}")

if __name__ == "__main__":
  
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
  
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
    
    df = load_data(DATA_PATH)
 
    overview = dataset_overview(df)
    print(overview_to_markdown(overview))
    


    README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
    save_report_to_file(df, README_PATH)

    
    data = prepare_pipeline_data(
        DATA_PATH, 
        sample_size=10000,
        verbose=True
    )
