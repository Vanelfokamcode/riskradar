# src/build_dataset.py
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH  = Path("data/riskradar.db")
OUT_PATH = Path("data/processed/dataset.parquet")

def build():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))

    df = conn.execute("""
        WITH bilans_clean AS (
            -- Dédoublonnage : garder le bilan le plus récent par siren+annee
            SELECT DISTINCT ON (siren, annee)
                siren, annee, denomination, date_cloture,
                code_activite,
                total_actif, actif_circulant, tresorerie,
                fonds_propres, dettes_lt, dettes_ct,
                ca, result_net
            FROM bilans
            WHERE total_actif IS NOT NULL
              AND total_actif > 0
              AND annee BETWEEN 2018 AND 2022
            ORDER BY siren, annee
        ),
        avec_target AS (
            SELECT
                b.*,
                -- Target : 1 si défaillance APRÈS le bilan (anti-leakage)
                CASE
                    WHEN d.siren IS NOT NULL
                     AND d.date_jugement > (b.date_cloture + INTERVAL '6 months')
                    THEN 1
                    ELSE 0
                END AS target,
                d.date_jugement,
                d.type_procedure
            FROM bilans_clean b
            LEFT JOIN defaillances d USING (siren)
        )
        SELECT * FROM avec_target
        ORDER BY annee, siren
    """).df()

    conn.close()

    print(f"Dataset total    : {len(df):,} lignes")
    print(f"Défauts (target=1): {df['target'].sum():,} ({df['target'].mean()*100:.2f}%)")
    print(f"Sains  (target=0): {(df['target']==0).sum():,}")
    print(f"\nRépartition par année :")
    print(df.groupby('annee')['target'].agg(['count','sum','mean']).round(3))

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSauvegardé : {OUT_PATH}")

if __name__ == "__main__":
    build()