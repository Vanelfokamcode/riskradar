import json
import duckdb
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/inpi")
DB_PATH  = Path("data/riskradar.db")

CODES = {
    "BJ": "total_actif",
    "BX": "actif_circulant",
    "BZ": "tresorerie",
    "CF": "fonds_propres",
    "DL": "dettes_lt",
    "DV": "dettes_ct",
    "FA": "ca",
    "FR": "ca_fr",
    "GF": "result_net",
}

FINANCIAL_COLS = ["total_actif", "actif_circulant", "tresorerie",
                  "fonds_propres", "dettes_lt", "dettes_ct",
                  "ca", "ca_fr", "result_net"]

def parse_montant(val: str) -> float:
    try:
        raw = float(val.strip()) if val and val.strip() else None
        return raw / 100.0 if raw is not None else None
    except:
        return None

def extract_liasses(liasses: list) -> dict:
    result = {col: None for col in FINANCIAL_COLS}
    for liasse in liasses:
        code = liasse.get("code")
        if code in CODES:
            val = parse_montant(liasse.get("m3") or liasse.get("m1", ""))
            result[CODES[code]] = val
    if result["ca"] is None and result["ca_fr"] is not None:
        result["ca"] = result["ca_fr"]
    return result

def parse_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    rows = []
    for entry in data:
        siren        = entry.get("siren")
        denomination = entry.get("denomination")
        bilan        = entry.get("bilanSaisi", {}).get("bilan", {})
        identite     = bilan.get("identite", {})
        date_cloture = identite.get("dateClotureExercice")
        code_activite= identite.get("codeActivite")

        try:
            annee = int(date_cloture[:4])
        except:
            continue

        liasses = []
        for page in bilan.get("detail", {}).get("pages", []):
            liasses.extend(page.get("liasses", []))

        financials = extract_liasses(liasses)
        rows.append({
            "siren": siren, "annee": annee,
            "denomination": denomination, "date_cloture": date_cloture,
            "code_activite": code_activite, **financials,
        })
    return rows

def ingest_all(limit_files: int = None):
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bilans (
            siren           VARCHAR,
            annee           INTEGER,
            denomination    VARCHAR,
            date_cloture    DATE,
            code_activite   VARCHAR,
            total_actif     DOUBLE,
            actif_circulant DOUBLE,
            tresorerie      DOUBLE,
            fonds_propres   DOUBLE,
            dettes_lt       DOUBLE,
            dettes_ct       DOUBLE,
            ca              DOUBLE,
            ca_fr           DOUBLE,
            result_net      DOUBLE,
            PRIMARY KEY (siren, annee)
        )
    """)

    files = sorted(RAW_DIR.glob("*.json"))
    if limit_files:
        files = files[:limit_files]

    total_rows = 0
    for i, path in enumerate(files):
        rows = parse_file(path)
        if not rows:
            continue

        df = pd.DataFrame(rows)
        conn.register("df_temp", df)
        conn.execute("""
            INSERT OR IGNORE INTO bilans
            SELECT siren, annee, denomination, date_cloture, code_activite,
                   total_actif, actif_circulant, tresorerie,
                   fonds_propres, dettes_lt, dettes_ct,
                   ca, ca_fr, result_net
            FROM df_temp
        """)
        conn.unregister("df_temp")

        total_rows += len(rows)
        if i % 100 == 0:
            print(f"[{i+1}/{len(files)}] {total_rows} bilans ingérés...")

    print(f"\nTerminé : {total_rows} bilans dans DuckDB")
    conn.close()

if __name__ == "__main__":
    ingest_all(limit_files=None)
