# 🎯 RiskRadar — Credit Risk Scoring Engine

> Moteur de scoring de risque de crédit sur données réelles françaises.  
> Pipeline DE complet : INPI × BODACC → DuckDB → XGBoost → SHAP → FastAPI → Streamlit

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.865-green)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)](https://fastapi.tiangolo.com)
[![Dagster](https://img.shields.io/badge/Dagster-Orchestration-purple)](https://dagster.io)

---

## Le problème

Les équipes Risques en banque scorent manuellement des milliers d'entreprises chaque année — calcul de ratios financiers, consultation des bases de défaillances, rédaction de notes d'analyse. RiskRadar automatise ce workflow de bout en bout.

---

## Les données

| Source | Contenu | Volume |
|--------|---------|--------|
| **INPI / RNE** | Bilans comptables (actif, passif, compte de résultat) | 6.3M bilans |
| **BODACC** | Défaillances officielles (sauvegarde, redressement, liquidation) | 33k procédures |

- Jointure sur le **SIREN** (identifiant universel des entreprises françaises)
- Target variable : défaut = 1 si procédure collective publiée au BODACC
- Split **chronologique** strict (train ≤ 2020 / val 2021 / test 2022) — anti data leakage

---

## Architecture
INPI SFTP ──┐
├─→ DuckDB ──→ Feature Engineering ──→ XGBoost ──→ FastAPI
BODACC API ─┘                                          │
SHAP ──→ Streamlit
│
Dagster (schedule mensuel)

---

## Modèle

| Modèle | AUC-ROC | Gini |
|--------|---------|------|
| Baseline Z-Score | 0.508 | 0.016 |
| Logistic Regression | 0.784 | 0.567 |
| **XGBoost** | **0.865** | **0.731** |

**Features** : 10 ratios financiers PCG (liquidité, endettement, rentabilité, activité) + Altman Z-Score adapté données privées.

**Explainabilité** : SHAP values individuelles par entreprise — conforme aux exigences Bâle III / RGPD.

---

## Stack technique

- **Ingestion** : Python, DuckDB, SFTP (paramiko)
- **Feature engineering** : pandas, numpy — ratios PCG, Altman Z-Score
- **Modèle** : XGBoost, scikit-learn, MLflow (experiment tracking)
- **Explainabilité** : SHAP (TreeExplainer, waterfall plots)
- **API** : FastAPI, Pydantic, uvicorn
- **Dashboard** : Streamlit, Plotly
- **Orchestration** : Dagster (schedule mensuel, alertes watchlist)

---

## Installation

```bash
git clone https://github.com/Vanelfokamcode/riskradar
cd riskradar
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Données** (accès INPI requis) :
```bash
# 1. Créer un compte sur data.inpi.fr
# 2. Demander accès SFTP RNE → Comptes annuels
# 3. Télécharger les fichiers dans data/raw/inpi/
python scripts/download_data.py --source bodacc --year 2022
```

**Pipeline complet** :
```bash
python src/ingest_inpi.py
python src/ingest_bodacc.py
python src/build_dataset.py
python src/features.py
python src/model.py
python src/explainer.py
```

**API** :
```bash
uvicorn src.api:app --reload --port 8000
# → http://localhost:8000/score/{siren}
# → http://localhost:8000/explain/{siren}
# → http://localhost:8000/history/{siren}
```

**Dashboard** :
```bash
streamlit run app.py
```

**Orchestration** :
```bash
dagster dev -f dagster_pipeline/jobs.py
```

---

## Les 3 scénarios de démo

**1. Dégradation progressive détectée automatiquement**
GET /history/865801401
→ 2018: 31% → 2019: 56% → 2020: 76% → 2021: 97% → 2022: 98%
Le modèle détecte la trajectoire de détresse 4 ans avant le signal BODACC.

**2. Explication individuelle SHAP**
GET /explain/865801401
→ Gearing LT = -2.73  SHAP=+1.55  ↑ risque
→ Solvabilité = 0.07  SHAP=+0.39  ↑ risque
Chaque score est explicable — conforme Bâle III.

**3. Comparaison sectorielle**
Recherche par secteur NAF/APE → score moyen du secteur vs entreprise cible.

---

## Structure du repo
riskradar/
├── src/
│   ├── ingest_inpi.py       # Ingestion bilans INPI
│   ├── ingest_bodacc.py     # Ingestion défaillances BODACC
│   ├── build_dataset.py     # Jointure + target variable
│   ├── features.py          # Feature engineering financier
│   ├── model.py             # XGBoost + MLflow
│   ├── explainer.py         # SHAP values
│   └── api.py               # FastAPI endpoints
├── dagster_pipeline/
│   └── jobs.py              # Orchestration + schedule mensuel
├── models/
│   └── xgboost_riskradar.json
├── app.py                   # Dashboard Streamlit
└── README.md

---

## Auteur

**Vanel Fokam**   
[GitHub](https://github.com/Vanelfokamcode) · vanelpouokamfokam@gmail.com
