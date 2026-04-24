# src/ingest_inpi.py
import json
import duckdb
from pathlib import Path

RAW_DIR = Path("data/raw/inpi")
DB_PATH = Path("data/riskradar.db")

# Codes PCG qu'on extrait
CODES = {
    "BJ": "total_actif",
    "BX": "actif_circulant",
    "BZ": "tresorerie",
    "CF": "fonds_propres",
    "DL": "dettes_lt",
    "DV": "dettes_ct",
    "FL": "ca",
    "GF": "result_net",
}

def parse_montant(val: str) -> float:
    """Convertit la string INPI '000000002801000' en float."""
    try:
        return float(val.strip()) if val.strip() else None
    except:
        return None

# Remplace extract_liasses et le bloc rows.append par ceci :

FINANCIAL_COLS = ["total_actif", "actif_circulant", "tresorerie",
                  "fonds_propres", "dettes_lt", "dettes_ct", "ca", "result_net"]

def extract_liasses(liasses: list) -> dict:
    result = {col: None for col in FINANCIAL_COLS}  # toutes les colonnes initialisées à None
    for liasse in liasses:
        code = liasse.get("code")
        if code in CODES:
            val = parse_montant(liasse.get("m3") or liasse.get("m1", ""))
            result[CODES[code]] = val
    return result

def parse_file(path: Path) -> list[dict]:
    """Parse un fichier JSON INPI et retourne une liste de bilans."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    # Chaque fichier est une liste d'entreprises
    if not isinstance(data, list):
        data = [data]

    for entry in data:
        siren = entry.get("siren")
        denomination = entry.get("denomination")

        bilan = entry.get("bilanSaisi", {}).get("bilan", {})
        identite = bilan.get("identite", {})
        date_cloture = identite.get("dateClotureExercice")
        code_activite = identite.get("codeActivite")  # NAF/APE

        # Extraire l'année depuis dateClotureExercice
        try:
            annee = int(date_cloture[:4])
        except:
            continue

        # Aplatir toutes les liasses de toutes les pages
        liasses = []
        for page in bilan.get("detail", {}).get("pages", []):
            liasses.extend(page.get("liasses", []))

        financials = extract_liasses(liasses)
        if not financials:
            continue

        rows.append({
            "siren": siren,
            "annee": annee,
            "denomination": denomination,
            "date_cloture": date_cloture,
            "code_activite": code_activite,
            **financials,
        })

    return rows

def ingest_all(limit_files: int = None):
    """Ingère tous les fichiers JSON dans DuckDB."""
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

        # Insert via DuckDB depuis une liste de dicts
        import pandas as pd
        df = pd.DataFrame(rows)

        # Remplace le INSERT OR IGNORE par ceci :
        conn.execute("""
            INSERT OR IGNORE INTO bilans
            SELECT
                siren, annee, denomination, date_cloture, code_activite,
                total_actif, actif_circulant, tresorerie,
                fonds_propres, dettes_lt, dettes_ct, ca, result_net
            FROM df
        """)
        total_rows += len(rows)
        if i % 100 == 0:
            print(f"[{i+1}/{len(files)}] {total_rows} bilans ingérés...")

    print(f"\nTerminé : {total_rows} bilans dans DuckDB")
    conn.close()

if __name__ == "__main__":
    ingest_all(limit_files=None)
