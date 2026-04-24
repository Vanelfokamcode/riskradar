# src/ingest_bodacc.py
import json
import duckdb
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/bodacc")
DB_PATH = Path("data/riskradar.db")

# On ne garde que les procédures collectives
FAMILLE_CIBLE = "collective"

# Mapping type de procédure → label propre
PROCEDURES = {
    "sauvegarde":             "SAUVEGARDE",
    "redressement":           "REDRESSEMENT",
    "liquidation":            "LIQUIDATION",
    "rétablissement":         "RETABLISSEMENT",
}

def extract_siren(registre) -> str | None:
    """Extrait le SIREN (9 chiffres) depuis le champ registre."""
    if not registre:
        return None
    candidates = registre if isinstance(registre, list) else [registre]
    for val in candidates:
        clean = str(val).replace(" ", "").strip()
        if len(clean) >= 9 and clean[:9].isdigit():
            return clean[:9]
    return None

def extract_procedure(jugement: str | dict | None) -> str | None:
    """Extrait le type de procédure depuis le champ jugement."""
    if not jugement:
        return None
    # Parfois c'est une string JSON, parfois un dict
    if isinstance(jugement, str):
        try:
            jugement = json.loads(jugement)
        except:
            jugement = {"complementJugement": jugement}
    
    complement = ""
    if isinstance(jugement, dict):
        complement = (
            jugement.get("complementJugement", "") or
            jugement.get("nature", "") or ""
        ).lower()
    
    for key, label in PROCEDURES.items():
        if key in complement:
            return label
    return "AUTRE"

def parse_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    rows = []
    for entry in data:
        # On ne garde que les procédures collectives
        if entry.get("familleavis") != FAMILLE_CIBLE:
            continue

        siren = extract_siren(entry.get("registre"))
        if not siren:
            continue

        date_parution = entry.get("dateparution")
        procedure = extract_procedure(entry.get("jugement"))

        rows.append({
            "siren":          siren,
            "date_jugement":  date_parution,
            "type_procedure": procedure,
            "denomination":   entry.get("commercant"),
            "tribunal":       entry.get("tribunal"),
        })

    return rows

def ingest_all():
    conn = duckdb.connect(str(DB_PATH))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS defaillances (
            siren          VARCHAR,
            date_jugement  DATE,
            type_procedure VARCHAR,
            denomination   VARCHAR,
            tribunal       VARCHAR
        )
    """)

    files = sorted(RAW_DIR.glob("bodacc_*.json"))
    total_rows = 0

    for path in files:
        rows = parse_file(path)
        if not rows:
            print(f"{path.name} — 0 procédures collectives trouvées")
            continue

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO defaillances SELECT * FROM df")
        total_rows += len(rows)
        print(f"{path.name} — {len(rows)} procédures insérées")

    # Dédoublonnage : garder la procédure la plus grave par SIREN
    conn.execute("""
        CREATE OR REPLACE TABLE defaillances AS
        SELECT DISTINCT ON (siren)
            siren, date_jugement, type_procedure, denomination, tribunal
        FROM defaillances
        ORDER BY siren,
            CASE type_procedure
                WHEN 'LIQUIDATION'    THEN 1
                WHEN 'REDRESSEMENT'   THEN 2
                WHEN 'SAUVEGARDE'     THEN 3
                WHEN 'RETABLISSEMENT' THEN 4
                ELSE 5
            END
    """)

    count = conn.execute("SELECT COUNT(*) FROM defaillances").fetchone()[0]
    print(f"\nTerminé : {count} entreprises en défaillance dans DuckDB")
    conn.close()

if __name__ == "__main__":
    ingest_all()