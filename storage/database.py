# storage/database.py
import duckdb
from pathlib import Path

DB_PATH = Path("data/riskradar.db")

class Database:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bilans (
                siren        VARCHAR,
                annee        INTEGER,
                denomination VARCHAR,
                date_cloture DATE,
                -- postes PCG bruts (on complète au chapitre 4)
                ca           DOUBLE,  -- chiffre d'affaires
                result_net   DOUBLE,  -- résultat net
                total_actif  DOUBLE,
                fonds_propres DOUBLE,
                dettes_lt    DOUBLE,
                dettes_ct    DOUBLE,
                tresorerie   DOUBLE,
                PRIMARY KEY (siren, annee)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS defaillances (
                siren          VARCHAR PRIMARY KEY,
                date_jugement  DATE,
                type_procedure VARCHAR  -- SAUVEGARDE / REDRESSEMENT / LIQUIDATION
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                siren         VARCHAR,
                annee         INTEGER,
                -- ratios (on complète au chapitre 7)
                current_ratio DOUBLE,
                roe           DOUBLE,
                debt_to_equity DOUBLE,
                altman_z      DOUBLE,
                target        INTEGER,  -- 0 = sain, 1 = défaut
                PRIMARY KEY (siren, annee)
            )
        """)

    def conn(self):
        return self._conn

    def query(self, sql: str, params=None):
        if params:
            return self._conn.execute(sql, params).df()
        return self._conn.execute(sql).df()

    def execute(self, sql: str, params=None):
        if params:
            self._conn.execute(sql, params)
        else:
            self._conn.execute(sql)

    def close(self):
        self._conn.close()