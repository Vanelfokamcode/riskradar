import dagster as dg
import subprocess
import pandas as pd
import xgboost as xgb

FEATURE_COLS = [
    'current_ratio', 'cash_ratio',
    'debt_to_equity', 'gearing', 'solvabilite',
    'roe', 'roa', 'marge_nette',
    'rotation_actif', 'altman_z',
]

WATCHLIST = ["865801401"]
CWD = "/home/vanel/riskradar"

def run_script(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", script],
        capture_output=True, text=True, cwd=CWD
    )

@dg.asset(description="Ingestion des bilans INPI")
def inpi_bilans():
    import duckdb
    conn = duckdb.connect(f"{CWD}/data/riskradar.db")
    count = conn.execute("SELECT COUNT(*) FROM bilans").fetchone()[0]
    conn.close()
    if count > 1_000_000:
        print(f"Skip — {count:,} bilans déjà présents")
        return dg.Output(int(count), metadata={"bilans": int(count), "source": "cache"})
    result = run_script("src/ingest_inpi.py")
    if result.returncode != 0:
        raise Exception(f"ingest_inpi failed:\n{result.stdout}\n{result.stderr}")
    return dg.Output(int(count), metadata={"bilans": int(count)})

@dg.asset(deps=["inpi_bilans"], description="Ingestion des défaillances BODACC")
def bodacc_defaillances():
    import duckdb
    conn = duckdb.connect(f"{CWD}/data/riskradar.db")
    count = conn.execute("SELECT COUNT(*) FROM defaillances").fetchone()[0]
    conn.close()
    if count > 1_000:
        print(f"Skip — {count:,} défaillances déjà présentes")
        return dg.Output(int(count), metadata={"defaillances": int(count), "source": "cache"})
    result = run_script("src/ingest_bodacc.py")
    if result.returncode != 0:
        raise Exception(f"ingest_bodacc failed:\n{result.stdout}\n{result.stderr}")
    return dg.Output(int(count), metadata={"defaillances": int(count)})

@dg.asset(deps=["inpi_bilans", "bodacc_defaillances"], description="Dataset annoté")
def dataset_annote():
    result = run_script("src/build_dataset.py")
    if result.returncode != 0:
        raise Exception(f"build_dataset failed:\n{result.stdout}\n{result.stderr}")
    df = pd.read_parquet(f"{CWD}/data/processed/dataset.parquet")
    return dg.Output(int(len(df)), metadata={
        "lignes": int(len(df)),
        "défauts": int(df['target'].sum()),
        "taux_défaut": float(round(df['target'].mean() * 100, 4)),
    })

@dg.asset(deps=["dataset_annote"], description="Feature engineering")
def features_engineered():
    result = run_script("src/features.py")
    if result.returncode != 0:
        raise Exception(f"features failed:\n{result.stdout}\n{result.stderr}")
    df = pd.read_parquet(f"{CWD}/data/processed/features.parquet")
    return dg.Output(int(len(df)), metadata={"lignes": int(len(df))})

@dg.asset(deps=["features_engineered"], description="XGBoost re-training")
def modele_xgboost():
    result = run_script("src/model.py")
    if result.returncode != 0:
        raise Exception(f"model failed:\n{result.stdout}\n{result.stderr}")
    return dg.Output(True, metadata={"status": "ok"})

@dg.asset(deps=["modele_xgboost"], description="SHAP values")
def shap_values():
    result = run_script("src/explainer.py")
    if result.returncode != 0:
        raise Exception(f"explainer failed:\n{result.stdout}\n{result.stderr}")
    return dg.Output(True, metadata={"status": "ok"})

@dg.asset(deps=["shap_values"], description="Alertes watchlist")
def watchlist_alertes():
    model = xgb.XGBClassifier()
    model.load_model(f"{CWD}/models/xgboost_riskradar.json")
    df = pd.read_parquet(f"{CWD}/data/processed/features.parquet")

    alertes = []
    for siren in WATCHLIST:
        rows = df[df['siren'] == siren]
        if rows.empty:
            continue
        latest = rows.sort_values('annee').iloc[-1]
        X = pd.DataFrame(
            [latest[FEATURE_COLS].fillna(0).values],
            columns=FEATURE_COLS
        )
        score = float(model.predict_proba(X)[:, 1][0])
        if score > 0.20:
            alertes.append({"siren": siren, "score_pd": float(round(score * 100, 2))})
            print(f"🚨 ALERTE — {siren} score={score*100:.2f}%")
        else:
            print(f"✅ OK — {siren} score={score*100:.2f}%")

    return dg.Output(int(len(alertes)), metadata={"alertes": int(len(alertes))})

riskradar_job = dg.define_asset_job(
    name="riskradar_pipeline",
    selection=[
        "inpi_bilans", "bodacc_defaillances", "dataset_annote",
        "features_engineered", "modele_xgboost", "shap_values", "watchlist_alertes"
    ]
)

riskradar_schedule = dg.ScheduleDefinition(
    job=riskradar_job,
    cron_schedule="0 6 1 * *",
    name="riskradar_monthly",
)

defs = dg.Definitions(
    assets=[
        inpi_bilans, bodacc_defaillances, dataset_annote,
        features_engineered, modele_xgboost, shap_values, watchlist_alertes,
    ],
    jobs=[riskradar_job],
    schedules=[riskradar_schedule],
)
