import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from typing import Optional, Tuple, Dict, Union

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from data_prep import prepare_pipeline_data, RANDOM_STATE
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)



_report_lines: List[str] = []


def log(msg: str = "", also_print: bool = True) -> None:

    if also_print:
        print(msg)
    _report_lines.append(msg)


def save_report(filename: str = "experiments.md") -> None:

    report_path = REPORTS_DIR / filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_report_lines))
    print(f" Report saved to {report_path}")


def clear_report() -> None:

    _report_lines.clear()




if __name__ == "__main__":
    
    log(" Train module initialized successfully.")
    log(f"Data path: {DATA_PATH}")
    log(f"Models directory: {MODELS_DIR}")
    log(f"Reports directory: {REPORTS_DIR}")
    log("All paths are valid." if DATA_PATH.exists() else "Data file not found!")
    save_report("setup_check.md")



    
def evaluate(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_proba: Optional[Union[np.ndarray, list]] = None,
    label: str = "",
    verbose: bool = True,
) -> Dict[str, Union[float, int, np.ndarray]]:

    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            results["roc_auc"] = auc
        except ValueError:
            results["roc_auc"] = None
    
    if verbose:
        prefix = f"{label} - " if label else ""
        log(f"\n{prefix}Evaluation Results:")
        log(f"  Accuracy : {acc:.4f}")
        log(f"  Precision: {prec:.4f}")
        log(f"  Recall   : {rec:.4f}")
        log(f"  F1-score : {f1:.4f}")
        if "roc_auc" in results and results["roc_auc"] is not None:
            log(f"  AUC-ROC  : {results['roc_auc']:.4f}")
        log(f"  Confusion Matrix:\n{cm}")
    
    return results


def plot_roc_curves(
    models_dict: Dict[str, object],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    show_random_line: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:

  
    valid_models = {
        name: model for name, model in models_dict.items()
        if hasattr(model, "predict_proba")
    }
    
    if not valid_models:
        log("No models support predict_proba(). Cannot plot ROC curves.")
        return None, None
    
    n_models = len(valid_models)
    n_cols = min(2, n_models) 
    n_rows = (n_models + n_cols - 1) // n_cols
    
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False
    )
    axes_flat = axes.flatten()
    
    
    for idx, (name, model) in enumerate(valid_models.items()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        ax.plot(fpr, tpr, linewidth=2,
                label=f'{name}\n(AUC = {auc:.3f})',
                color=plt.cm.tab10(idx % 10))
        
        if show_random_line:
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
                    label='Random (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        
        ax.text(0.05, 0.05, f'AUC = {auc:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
   
    for idx in range(len(valid_models), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    
    fig.suptitle('ROC Curves - Credit Card Fraud Detection',
                 fontsize=14, fontweight='bold', y=1.02)
    
   
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # ذخیره
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        log(f"ROC curves saved to: {save_path}")
    
    plt.show()
    return fig, axes


def save_report(
    filename: str = "experiments.md",
    add_timestamp: bool = False,
    backup_existing: bool = True,
) -> Optional[Path]:

    report_path = REPORTS_DIR / filename

  
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = report_path.stem
        suffix = report_path.suffix
        report_path = REPORTS_DIR / f"{stem}_{timestamp}{suffix}"

   
    if backup_existing and report_path.exists():
        backup_path = report_path.with_suffix(".backup")
        try:
            report_path.rename(backup_path)
            log(f"Existing report backed up to: {backup_path}")
        except Exception as e:
            log(f"Could not backup existing report: {e}")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            # Add a header with timestamp
            f.write(f"# Experiment Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            # Write all collected messages
            f.write("\n".join(_report_lines))
        
        log(f"Report saved to: {report_path}")
        return report_path

    except PermissionError:
        log(f"Permission denied: Cannot write to {report_path}")
    except OSError as e:
        log(f"OS error while saving report: {e}")
    except Exception as e:
        log(f"Unexpected error while saving report: {e}")

    return None




def save_model(
    model: object,
    scaler: object,
    model_filename: str = "model.pkl",
    scaler_filename: str = "scaler.pkl",
    compress: Union[bool, int] = False,
) -> Tuple[Optional[Path], Optional[Path]]:

    MODELS_DIR.mkdir(exist_ok=True)

  
    model_path = MODELS_DIR / model_filename
    scaler_path = MODELS_DIR / scaler_filename

    saved_model_path = None
    saved_scaler_path = None

  
    try:
        joblib.dump(model, model_path, compress=compress)
        log(f"Model saved to: {model_path}")
        saved_model_path = model_path
    except (PermissionError, OSError, Exception) as e:
        log(f"Failed to save model: {e}")
   
    
    try:
        joblib.dump(scaler, scaler_path, compress=compress)
        log(f"Scaler saved to: {scaler_path}")
        saved_scaler_path = scaler_path
    except (PermissionError, OSError, Exception) as e:
        log(f"Failed to save scaler: {e}")

    
    return saved_model_path, saved_scaler_path


def main():
    log("# Credit Card Fraud Detection - Experiment Report")
    log(f"\nGenerated by `src/train.py`. Random state = {RANDOM_STATE}.\n")
    

# ================ Phase 1-2: Data Preparation ================
log("\n" + "=" * 60)
log("## Phase 1-2: Data Preparation & Preprocessing")
log("=" * 60)

try:
    data = prepare_pipeline_data(
        csv_path=DATA_PATH,
        test_size=0.2,
        random_state=RANDOM_STATE,
        sample_size=None,
        verbose=True
    )
except Exception as e:
    log(f"Error during data preparation: {e}")
    log("Training pipeline aborted.")
    raise


X_train_raw = data["X_train"]
X_test_raw = data["X_test"]
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
y_train = data["y_train"]
y_test = data["y_test"]
scaler = data["scaler"]


train_fraud_ratio = y_train.mean()
test_fraud_ratio = y_test.mean()


log(f"Data loaded successfully.")
log(f"  • Training samples: {len(X_train_raw):,}")
log(f"  • Test samples:     {len(X_test_raw):,}")
log(f"  • Features:         {X_train_raw.shape[1]}")
log(f"  • Fraud ratio (train): {train_fraud_ratio:.4%}  ({y_train.sum():,} cases)")
log(f"  • Fraud ratio (test):  {test_fraud_ratio:.4%}  ({y_test.sum():,} cases)")

if X_train_scaled is not None:
    log(f"  • Scaled features mean: {X_train_scaled.mean().mean():.4f}")
    log(f"  • Scaled features std:  {X_train_scaled.std().mean():.4f}")

log("")  


log("-" * 40)
log("Splitting training data into Train/Validation (80/20) for timing measurements")
log("-" * 40)

X_train_s, X_val_s, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train
)

log(f"Train size: {len(X_train_s):,}  |  Validation size: {len(X_val_s):,}")
log(f"Train fraud ratio: {y_train_split.mean():.4%}  |  Validation fraud ratio: {y_val_split.mean():.4%}")
log(f"Test size (untouched): {len(X_test_scaled):,}")
log(f"Test fraud ratio: {y_test.mean():.4%}\n")


log("\n" + "=" * 60)
log("## Phase 3: Model Training & Comparison (with Timing)")
log("=" * 60)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,              
        class_weight='balanced',   
        random_state=RANDOM_STATE,
        solver='liblinear',         
        C=1.0,                     
    ),
    
    "KNN (k=5, distance)": KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',         
        metric='euclidean',        
        n_jobs=-1,                  
    ),
    
    "Decision Tree (depth=10)": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',   
        max_depth=10,               
        min_samples_split=20,       
        min_samples_leaf=10,        
        criterion='gini',           #
    ),
}


log("Models defined with the following configurations:")
for name, model in models.items():
   
    if hasattr(model, 'get_params'):
        params = model.get_params()
        
        important_params = {
            k: v for k, v in params.items()
            if k in ['max_iter', 'class_weight', 'n_neighbors', 'weights',
                     'max_depth', 'min_samples_split', 'criterion']
        }
        log(f"  • {name}: {important_params}")
log("")


fitted_models = {}
test_results = {}

for name, model in models.items():
    log(f"\n--- {name} ---")
    
    try:
        
        train_start = time.time()
        model.fit(X_train_s, y_train_split)
        train_time = time.time() - train_start
        
      
        pred_start_val = time.time()
        y_pred_val = model.predict(X_val_s)
        pred_time_val = time.time() - pred_start_val
        
       
        pred_start_test = time.time()
        y_pred_test = model.predict(X_test_scaled)
        pred_time_test = time.time() - pred_start_test
        
       
        y_proba_test = None
        if hasattr(model, "predict_proba"):
            y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        
        
        metrics = evaluate(
            y_true=y_test,
            y_pred=y_pred_test,
            y_proba=y_proba_test,
            label=f"{name} (Test Set)",
            verbose=True
        )
        
       
        metrics.update({
            "train_time": train_time,
            "predict_time_val": pred_time_val,
            "predict_time_test": pred_time_test
        })
        
        test_results[name] = metrics
        fitted_models[name] = model
        
        log(f"⏱Train: {train_time:.2f}s | Predict (Val): {pred_time_val:.2f}s | (Test): {pred_time_test:.2f}s")
    
    except Exception as e:
        log(f"Error training {name}: {e}")
        continue
    
    log("")  



log("\n" + "=" * 60)
log("Summary of Test Results (with Timing & AUC)")
log("=" * 60)
log(f"{'Model':<30} {'F1':<8} {'AUC':<8} {'Train (s)':<10} {'Predict (s)':<10}")
log("-" * 70)

best_f1 = -1
best_model = None

for name, metrics in test_results.items():
   
    auc_val = metrics.get('roc_auc')
    auc_str = f"{auc_val:.4f}" if auc_val is not None else "-"
    
    
    pred_time = metrics.get('predict_time_test', metrics.get('predict_time_val', 0))
    
    log(f"{name:<30} {metrics['f1']:.4f}   {auc_str:<8} {metrics['train_time']:.3f}     {pred_time:.3f}")
    
   
    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_model = name

log("-" * 70)
log(f"Best model based on F1: {best_model} (F1 = {best_f1:.4f})")
log("=" * 60 + "\n")


# ================ ROC-AUC Analysis ================
log("\n" + "=" * 60)
log("## ROC-AUC Analysis")
log("=" * 60)


models_with_proba = {
    name: model for name, model in fitted_models.items()
    if hasattr(model, "predict_proba")
}

if models_with_proba:
  
    plot_path = REPORTS_DIR / "roc_curves.png"
    plot_roc_curves(
        models_dict=models_with_proba,
        X_test=X_test_scaled,
        y_test=y_test,
        save_path=str(plot_path)  
    )
    
    
    log("\nAUC-ROC Scores Summary:")
    log(f"{'Model':<30} {'AUC-ROC':<10}")
    log("-" * 42)
    
    for name, metrics in test_results.items():
        auc_val = metrics.get('roc_auc')  
        if auc_val is not None:
            log(f"{name:<30} {auc_val:.4f}")
        else:
            log(f"{name:<30} {'N/A':<10}")
else:
    log("No models support predict_proba(). Skipping ROC curve plotting.")

log("=" * 60 + "\n")

# ================ Best Model Selection ================
log("\n" + "=" * 60)
log("## Best Model Selection")
log("=" * 60)


best_model_name = None
best_f1 = -1
best_auc = -1

for name, metrics in test_results.items():
    current_f1 = metrics['f1']
    current_auc = metrics.get('roc_auc', 0)
    
    
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_auc = current_auc
        best_model_name = name
    
    elif current_f1 == best_f1 and current_auc > best_auc:
        best_auc = current_auc
        best_model_name = name


if best_model_name:
    best_model = fitted_models[best_model_name]
    log(f"Best model selected: **{best_model_name}**")
    log(f"   • F1-score:  {test_results[best_model_name]['f1']:.4f}")
    log(f"   • AUC-ROC:   {test_results[best_model_name].get('roc_auc', 'N/A')}")
    log(f"   • Train time: {test_results[best_model_name]['train_time']:.3f}s")
    log(f"   • Predict time (Test): {test_results[best_model_name].get('predict_time_test', 0):.3f}s")
else:
    log("No model found!")

log("=" * 60 + "\n")


# ================ Cross Validation (5-Fold Stratified) ================
log("\n" + "=" * 60)
log("## Cross Validation (5-Fold Stratified)")
log("=" * 60)


cv_models = ["Logistic Regression", "KNN", "Decision Tree"]


available_cv_models = [name for name in cv_models if name in fitted_models]
if not available_cv_models:
    log("No models available for Cross-Validation. Skipping CV.")
else:
    log("| Model | Precision (mean±std) | Recall (mean±std) | F1 (mean±std) |")
    log("|-------|---------------------|-------------------|---------------|")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["precision", "recall", "f1"]
    cv_results = {}

    for name in available_cv_models:
        model = fitted_models[name]  
        
        cv_start = time.time()
        cv = cross_validate(
            model, X_train_scaled, y_train,  
            cv=skf, scoring=scoring, n_jobs=-1
        )
        cv_time = time.time() - cv_start

        mean_prec = cv["test_precision"].mean()
        std_prec = cv["test_precision"].std()
        mean_rec = cv["test_recall"].mean()
        std_rec = cv["test_recall"].std()
        mean_f1 = cv["test_f1"].mean()
        std_f1 = cv["test_f1"].std()

        cv_results[name] = {
            "precision": mean_prec,
            "precision_std": std_prec,
            "recall": mean_rec,
            "recall_std": std_rec,
            "f1": mean_f1,
            "f1_std": std_f1,
            "cv_time": cv_time
        }

        log(f"| {name} | {mean_prec:.4f} ± {std_prec:.4f} | {mean_rec:.4f} ± {std_rec:.4f} | {mean_f1:.4f} ± {std_f1:.4f} |")
    
  
    if cv_results:
        best_cv_model = max(cv_results, key=lambda k: cv_results[k]['f1'])
        log("\n" + "=" * 60)
        log(f"Best model from CV: **{best_cv_model}** (CV F1 = {cv_results[best_cv_model]['f1']:.4f})")
        log(f"CV time: {cv_results[best_cv_model]['cv_time']:.3f}s")

log("=" * 60 + "\n")

# ================ Comparison: Test vs CV ================
if cv_results and test_results:
    log("\n" + "=" * 60)
    log("## Comparison: Test Results vs CV Results")
    log("=" * 60)
    log(f"{'Model':<30} {'Test F1':<12} {'CV F1':<12} {'Difference':<12}")
    log("-" * 70)
    
    for name in available_cv_models:
        if name in test_results and name in cv_results:
            test_f1 = test_results[name]['f1']
            cv_f1 = cv_results[name]['f1']
            diff = abs(test_f1 - cv_f1)
            log(f"{name:<30} {test_f1:.4f}      {cv_f1:.4f}      {diff:.4f}")
    
    log("=" * 60 + "\n")

# ================ Time Analysis ================
if test_results:
    log("\n" + "=" * 60)
    log("## Time Analysis")
    log("=" * 60)
    
    fastest_train = min(test_results, key=lambda k: test_results[k]['train_time'])
    
    fastest_predict = min(
        test_results,
        key=lambda k: test_results[k].get('predict_time_test', test_results[k].get('predict_time_val', float('inf')))
    )
    
    log(f"Fastest to train: **{fastest_train}** ({test_results[fastest_train]['train_time']:.3f}s)")
    log(f"Fastest to predict: **{fastest_predict}** ({test_results[fastest_predict].get('predict_time_test', test_results[fastest_predict].get('predict_time_val', 0)):.3f}s)")
    
   
    log("\n**Trade-off Analysis:**")
    log("- Logistic Regression: Best balance of speed and accuracy")
    log("- KNN: Slowest prediction (distance calculation to all training points)")
    log("- Decision Tree: Fast prediction, moderate training")
    if "MLP (Bonus)" in test_results:
        log("- MLP (Bonus): Slowest training, fast prediction")
    
    log("=" * 60 + "\n")

# ================ Final Model Selection ================
if test_results:
    log("\n" + "=" * 60)
    log("## Final Model Selection")
    log("=" * 60)
    
  
    best_model_name = None
    best_f1 = -1
    best_auc = -1
    
    for name, metrics in test_results.items():
        current_f1 = metrics['f1']
        current_auc = metrics.get('roc_auc', 0) or 0
        
        if current_f1 > best_f1 or (current_f1 == best_f1 and current_auc > best_auc):
            best_f1 = current_f1
            best_auc = current_auc
            best_model_name = name
    
    if best_model_name and best_model_name in fitted_models:
        final_model = fitted_models[best_model_name]
        log(f"Final model selected: **{best_model_name}**")
        log(f"   • F1 (Test):   {test_results[best_model_name]['f1']:.4f}")
        log(f"   • AUC (Test):  {test_results[best_model_name].get('roc_auc', 'N/A')}")
        log(f"   • Train time:  {test_results[best_model_name]['train_time']:.3f}s")
        log(f"   • Predict time: {test_results[best_model_name].get('predict_time_test', test_results[best_model_name].get('predict_time_val', 0)):.3f}s")
    else:
        log("No suitable model found for final selection.")

# ================ Save Results ================
log("\n" + "=" * 60)
log("## Saving Results")
log("=" * 60)


save_report()


if best_model_name and best_model_name in fitted_models:
    final_model = fitted_models[best_model_name]
    model_filename = best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    model_path, scaler_path = save_model(
        model=final_model,
        scaler=scaler,
        model_filename=f"{model_filename}.pkl",
        scaler_filename="scaler.pkl",
        compress=3
    )
    
    if model_path and scaler_path:
        log(f"Model saved: {model_path}")
        log(f"Scaler saved: {scaler_path}")
    else:
        log("Model or scaler could not be saved.")

log("\nTraining pipeline completed successfully!")
log("=" * 60 + "\n")

# ============================================================================
if __name__ == "__main__":
    main()
    
    
