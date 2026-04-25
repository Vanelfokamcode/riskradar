# src/build_dataset.py
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH  = Path("data/riskradar.db")
OUT_PATH = Path("data/processed/dataset.parquet")

# src/build_dataset.py — remplace la fonction build() entière :
def build():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))

    # Tout se passe dans DuckDB — zéro chargement en RAM
    conn.execute(f"""
        COPY (
            WITH bilans_clean AS (
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
        ) TO '{OUT_PATH}' (FORMAT PARQUET)
    """)

    # Stats légères
    stats = conn.execute(f"""
        SELECT 
            COUNT(*) as total,
            SUM(target) as defauts,
            AVG(target)*100 as taux
        FROM read_parquet('{OUT_PATH}')
    """).fetchone()

    print(f"Dataset total     : {stats[0]:,} lignes")
    print(f"Défauts (target=1): {int(stats[1]):,} ({stats[2]:.2f}%)")

    # Stats par année
    df_stats = conn.execute(f"""
        SELECT annee, COUNT(*) as count, SUM(target) as sum, AVG(target) as mean
        FROM read_parquet('{OUT_PATH}')
        GROUP BY annee ORDER BY annee
    """).df()
    print(f"\nRépartition par année :")
    print(df_stats.round(3))
    print(f"\nSauvegardé : {OUT_PATH}")
    conn.close()
    
if __name__ == "__main__":
    build()