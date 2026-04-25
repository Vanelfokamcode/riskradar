import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

IN_PATH  = Path("data/processed/features.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    'current_ratio', 'cash_ratio',
    'debt_to_equity', 'gearing', 'solvabilite',
    'roe', 'roa', 'marge_nette',
    'rotation_actif', 'altman_z',
]

def split_chronologique(df: pd.DataFrame):
    """Split temporel strict — jamais aléatoire en finance."""
    train = df[df['annee'] <= 2020].copy()
    val   = df[df['annee'] == 2021].copy()
    test  = df[df['annee'] == 2022].copy()

    print(f"Train : {len(train):,} lignes ({train['target'].sum()} défauts)")
    print(f"Val   : {len(val):,} lignes ({val['target'].sum()} défauts)")
    print(f"Test  : {len(test):,} lignes ({test['target'].sum()} défauts)")
    return train, val, test

def get_XY(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    y = df['target']
    return X, y

def eval_model(name: str, y_true, y_pred_proba):
    auc  = roc_auc_score(y_true, y_pred_proba)
    ap   = average_precision_score(y_true, y_pred_proba)
    gini = 2 * auc - 1
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Gini     : {gini:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    return {"auc": auc, "gini": gini, "avg_precision": ap}

def baseline_zscore(train, test):
    """Baseline : règle simple basée sur altman_z seul."""
    # Plus le Z est bas, plus le risque est élevé
    # On inverse pour avoir une probabilité de défaut
    _, y_test = get_XY(test)
    z = test['altman_z'].fillna(0)
    # Normalise entre 0 et 1 (inversé)
    z_norm = 1 - (z - z.min()) / (z.max() - z.min() + 1e-9)
    return eval_model("Baseline Z-Score", y_test, z_norm)

def train_logistic(train, test):
    """Logistic Regression comme premier modèle ML."""
    X_train, y_train = get_XY(train)
    X_test,  y_test  = get_XY(test)

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:, 1]
    return eval_model("Logistic Regression", y_test, proba)

def train_xgboost(train, val, test):
    """XGBoost avec gestion du déséquilibre."""
    X_train, y_train = get_XY(train)
    X_val,   y_val   = get_XY(val)
    X_test,  y_test  = get_XY(test)

    # Ratio négatifs/positifs pour scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"\nscale_pos_weight = {spw:.1f} ({neg} sains / {pos} défauts)")

    with mlflow.start_run(run_name="xgboost_v1"):
        params = {
            "n_estimators":     500,
            "max_depth":        4,
            "learning_rate":    0.05,
            "scale_pos_weight": spw,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "random_state":     42,
            "eval_metric":      "auc",
            "early_stopping_rounds": 30,
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        proba = model.predict_proba(X_test)[:, 1]
        metrics = eval_model("XGBoost", y_test, proba)
        mlflow.log_metrics(metrics)

        # Sauvegarde
        model.save_model(str(MODEL_DIR / "xgboost_riskradar.json"))
        mlflow.xgboost.log_model(model, "model")
        print(f"\nModèle sauvegardé : {MODEL_DIR}/xgboost_riskradar.json")

    return model, metrics

if __name__ == "__main__":
    mlflow.set_experiment("riskradar")

    df = pd.read_parquet(IN_PATH)
    print(f"Dataset : {len(df):,} lignes, {df['target'].sum()} défauts")

    train, val, test = split_chronologique(df)

    print("\n--- BASELINE ---")
    baseline_zscore(train, test)

    print("\n--- LOGISTIC REGRESSION ---")
    train_logistic(train, test)

    print("\n--- XGBOOST ---")
    model, metrics = train_xgboost(train, val, test)
