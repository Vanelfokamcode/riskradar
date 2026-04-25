import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

IN_PATH   = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/xgboost_riskradar.json")
OUT_DIR   = Path("data/processed")

FEATURE_COLS = [
    'current_ratio', 'cash_ratio',
    'debt_to_equity', 'gearing', 'solvabilite',
    'roe', 'roa', 'marge_nette',
    'rotation_actif', 'altman_z',
]

FEATURE_LABELS = {
    'current_ratio':  'Ratio de liquidité',
    'cash_ratio':     'Ratio de trésorerie',
    'debt_to_equity': 'Dette / Fonds propres',
    'gearing':        'Gearing LT',
    'solvabilite':    'Solvabilité',
    'roe':            'ROE',
    'roa':            'ROA',
    'marge_nette':    'Marge nette',
    'rotation_actif': 'Rotation actifs',
    'altman_z':       'Altman Z-Score',
}

def load_model_and_data():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    df = pd.read_parquet(IN_PATH)
    # Test set = 2022
    test = df[df['annee'] == 2022].copy()
    X_test = test[FEATURE_COLS].fillna(0)

    return model, test, X_test

def compute_shap_values(model, X_test):
    print("Calcul des SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"SHAP values calculées : {shap_values.shape}")
    return explainer, shap_values

def plot_global_importance(shap_values, X_test):
    """Feature importance globale — mean |SHAP|."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=[FEATURE_LABELS[c] for c in FEATURE_COLS],
        plot_type="bar",
        show=False
    )
    plt.title("Importance globale des features (mean |SHAP|)", fontsize=14)
    plt.tight_layout()
    out = OUT_DIR / "shap_global_importance.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvegardé : {out}")

def plot_beeswarm(shap_values, X_test):
    """Beeswarm — distribution complète des SHAP values."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=[FEATURE_LABELS[c] for c in FEATURE_COLS],
        show=False
    )
    plt.title("Distribution des SHAP values par feature", fontsize=14)
    plt.tight_layout()
    out = OUT_DIR / "shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvegardé : {out}")

def explain_individual(model, explainer, shap_values, test, X_test, siren: str = None):
    """Explication individuelle — waterfall plot pour une entreprise."""
    if siren:
        idx_list = test[test['siren'] == siren].index.tolist()
        if not idx_list:
            print(f"SIREN {siren} non trouvé dans le test set")
            return
        idx = idx_list[0]
        pos = test.index.get_loc(idx)
    else:
        # Prend l'entreprise avec le score de risque le plus élevé
        proba = model.predict_proba(X_test)[:, 1]
        pos = np.argmax(proba)
        siren = test.iloc[pos]['siren']

    row = test.iloc[pos]
    proba_val = model.predict_proba(X_test.iloc[[pos]])[:, 1][0]

    print(f"\n--- Entreprise la plus risquée ---")
    print(f"SIREN      : {siren}")
    print(f"Dénomination: {row.get('denomination', 'N/A')}")
    print(f"Année      : {row['annee']}")
    print(f"Score PD   : {proba_val:.4f} ({proba_val*100:.2f}%)")
    print(f"Target réel: {int(row['target'])}")

    print(f"\nTop features :")
    sv = shap_values[pos]
    feature_impact = sorted(
        zip(FEATURE_COLS, sv, X_test.iloc[pos]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feat, sv_val, feat_val in feature_impact[:5]:
        direction = "↑ risque" if sv_val > 0 else "↓ risque"
        print(f"  {FEATURE_LABELS[feat]:<25} = {feat_val:8.3f}  SHAP={sv_val:+.4f}  {direction}")

    # Waterfall plot
    plt.figure(figsize=(10, 6))
    shap_exp = shap.Explanation(
        values=shap_values[pos],
        base_values=explainer.expected_value,
        data=X_test.iloc[pos].values,
        feature_names=[FEATURE_LABELS[c] for c in FEATURE_COLS]
    )
    shap.plots.waterfall(shap_exp, show=False)
    plt.title(f"Explication du score — SIREN {siren} (PD={proba_val*100:.2f}%)", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"shap_waterfall_{siren}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nWaterfall sauvegardé : {out}")

def save_shap_values(shap_values, test, X_test):
    """Sauvegarde les SHAP values pour l'API."""
    df_shap = pd.DataFrame(
        shap_values,
        columns=[f"shap_{c}" for c in FEATURE_COLS],
        index=X_test.index
    )
    df_shap['siren'] = test['siren'].values
    df_shap['annee'] = test['annee'].values
    df_shap['proba'] = model.predict_proba(X_test)[:, 1]

    out = OUT_DIR / "shap_values.parquet"
    df_shap.to_parquet(out, index=False)
    print(f"SHAP values sauvegardées : {out}")

if __name__ == "__main__":
    model, test, X_test = load_model_and_data()
    explainer, shap_values = compute_shap_values(model, X_test)

    print("\n=== IMPORTANCE GLOBALE ===")
    plot_global_importance(shap_values, X_test)

    print("\n=== BEESWARM ===")
    plot_beeswarm(shap_values, X_test)

    print("\n=== EXPLICATION INDIVIDUELLE ===")
    explain_individual(model, explainer, shap_values, test, X_test)

    print("\n=== SAUVEGARDE SHAP VALUES ===")
    save_shap_values(shap_values, test, X_test)
