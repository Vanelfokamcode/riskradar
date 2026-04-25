from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

app = FastAPI(
    title="RiskRadar API",
    description="Moteur de scoring de risque de crédit — données INPI/BODACC",
    version="1.0.0"
)

# --- Chargement au démarrage ---
MODEL_PATH   = Path("models/xgboost_riskradar.json")
FEATURES_PATH = Path("data/processed/features.parquet")
SHAP_PATH    = Path("data/processed/shap_values.parquet")

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

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

df_features = pd.read_parquet(FEATURES_PATH)
df_shap     = pd.read_parquet(SHAP_PATH)

# --- Schemas ---
class ScoreResponse(BaseModel):
    siren:        str
    denomination: str
    annee:        int
    score_pd:     float
    score_pct:    float
    zone:         str
    altman_z:     float

class ExplainResponse(BaseModel):
    siren:     str
    annee:     int
    score_pd:  float
    top_features: list[dict]

class HistoryPoint(BaseModel):
    annee:    int
    score_pd: float
    zone:     str

class HistoryResponse(BaseModel):
    siren:        str
    denomination: str
    history:      list[HistoryPoint]

# --- Helpers ---
def get_zone(score: float) -> str:
    if score < 0.05:
        return "saine"
    elif score < 0.20:
        return "surveillance"
    else:
        return "distress"

def get_latest(siren: str) -> pd.Series:
    rows = df_features[df_features['siren'] == siren]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SIREN {siren} non trouvé")
    return rows.sort_values('annee').iloc[-1]

def score_row(row: pd.Series) -> float:
    X = pd.DataFrame([row[FEATURE_COLS].fillna(0).values], columns=FEATURE_COLS)
    return float(model.predict_proba(X)[:, 1][0])

# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "service": "RiskRadar API v1.0"}

@app.get("/score/{siren}", response_model=ScoreResponse)
def score(siren: str):
    """Retourne le score de risque de défaut (PD) pour un SIREN."""
    row   = get_latest(siren)
    pd_score = score_row(row)
    return ScoreResponse(
        siren        = siren,
        denomination = str(row.get('denomination', 'N/A')),
        annee        = int(row['annee']),
        score_pd     = round(pd_score, 6),
        score_pct    = round(pd_score * 100, 2),
        zone         = get_zone(pd_score),
        altman_z     = round(float(row.get('altman_z', 0)), 3),
    )

@app.get("/explain/{siren}", response_model=ExplainResponse)
def explain(siren: str):
    """Retourne les features SHAP qui expliquent le score."""
    row = get_latest(siren)
    pd_score = score_row(row)

    # Cherche les SHAP values précalculées
    shap_row = df_shap[df_shap['siren'] == siren]
    if shap_row.empty:
        # Calcule à la volée si pas dans le test set
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(model)
        X = pd.DataFrame([row[FEATURE_COLS].fillna(0).values], columns=FEATURE_COLS)
        sv = explainer.shap_values(X)[0]
        shap_vals = dict(zip(FEATURE_COLS, sv))
        feat_vals = dict(zip(FEATURE_COLS, X.iloc[0]))
    else:
        shap_row = shap_row.sort_values('annee').iloc[-1]
        shap_vals = {c: float(shap_row[f"shap_{c}"]) for c in FEATURE_COLS}
        feat_vals = {c: float(row.get(c, 0)) for c in FEATURE_COLS}

    top = sorted(
        [
            {
                "feature":    c,
                "label":      FEATURE_LABELS[c],
                "value":      round(feat_vals[c], 4),
                "shap":       round(shap_vals[c], 4),
                "direction":  "↑ risque" if shap_vals[c] > 0 else "↓ risque",
            }
            for c in FEATURE_COLS
        ],
        key=lambda x: abs(x['shap']),
        reverse=True
    )[:5]

    return ExplainResponse(
        siren       = siren,
        annee       = int(row['annee']),
        score_pd    = round(pd_score, 6),
        top_features= top,
    )

@app.get("/history/{siren}", response_model=HistoryResponse)
def history(siren: str):
    """Retourne l'évolution du score sur toutes les années disponibles."""
    rows = df_features[df_features['siren'] == siren]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SIREN {siren} non trouvé")

    denomination = str(rows.iloc[-1].get('denomination', 'N/A'))
    hist = []
    for _, row in rows.sort_values('annee').iterrows():
        pd_score = score_row(row)
        hist.append(HistoryPoint(
            annee    = int(row['annee']),
            score_pd = round(pd_score, 6),
            zone     = get_zone(pd_score),
        ))

    return HistoryResponse(
        siren        = siren,
        denomination = denomination,
        history      = hist,
    )

@app.get("/search/{query}")
def search(query: str, limit: int = 10):
    """Recherche par nom d'entreprise (contient)."""
    mask = df_features['denomination'].str.contains(
        query, case=False, na=False
    )
    results = df_features[mask][['siren','denomination','annee']]\
        .drop_duplicates('siren')\
        .head(limit)

    return {
        "query":   query,
        "count":   len(results),
        "results": results.to_dict(orient='records')
    }
