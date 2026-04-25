import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = Path("data/processed/dataset.parquet")
OUT_PATH = Path("data/processed/features.parquet")

RATIO_COLS = [
    'current_ratio', 'cash_ratio',
    'debt_to_equity', 'gearing', 'solvabilite',
    'roe', 'roa', 'marge_nette',
    'rotation_actif', 'altman_z',
]

def safe_div(a, b):
    return pd.Series(
        np.where((b == 0) | b.isna(), np.nan, a / b),
        index=a.index
    )

def clean_chunk(df):
    df = df[df['total_actif'] >= 1000].copy()
    mask = df['result_net'].notna() & df['total_actif'].notna()
    df = df[~mask | (df['result_net'].abs() <= df['total_actif'] * 10)]
    mask2 = df['ca'].notna() & df['total_actif'].notna()
    df = df[~mask2 | (df['ca'] <= df['total_actif'] * 100)]
    return df

def compute_ratios(df):
    dt = df['dettes_lt'].fillna(0) + df['dettes_ct'].fillna(0)
    df['current_ratio']  = safe_div(df['actif_circulant'], df['dettes_ct'])
    df['cash_ratio']     = safe_div(df['tresorerie'],      df['dettes_ct'])
    df['debt_to_equity'] = safe_div(dt,                    df['fonds_propres'])
    df['gearing']        = safe_div(df['dettes_lt'],       df['fonds_propres'])
    df['solvabilite']    = safe_div(df['fonds_propres'],   df['total_actif'])
    df['roe']            = safe_div(df['result_net'],      df['fonds_propres'])
    df['roa']            = safe_div(df['result_net'],      df['total_actif'])
    df['marge_nette']    = safe_div(df['result_net'],      df['ca'])
    df['rotation_actif'] = safe_div(df['ca'],              df['total_actif'])
    return df

def compute_altman_z(df):
    ta = df['total_actif'].replace(0, np.nan)
    dt = (df['dettes_lt'].fillna(0) + df['dettes_ct'].fillna(0)).replace(0, np.nan)
    X1 = safe_div(df['actif_circulant'].fillna(0) - df['dettes_ct'].fillna(0), ta).clip(-5, 5)
    X2 = safe_div(df['result_net'].fillna(0), ta).clip(-5, 5)
    X4 = safe_div(df['fonds_propres'].fillna(0), dt).clip(-5, 5)
    X5 = safe_div(df['ca'].fillna(0), ta).clip(0, 10)
    df['altman_z'] = (6.56 * X1 + 1.05 * X4 + 1.0 * X5).clip(-30, 30)
    df['altman_zone'] = pd.cut(
        df['altman_z'],
        bins=[-np.inf, 1.23, 2.90, np.inf],
        labels=['distress', 'grise', 'saine']
    )
    return df

def compute_winsor_bounds():
    """Calcule les bornes sur un échantillon de 200k lignes."""
    df = pd.read_parquet(IN_PATH).sample(n=200_000, random_state=42)
    df = clean_chunk(df)
    df = compute_ratios(df)
    df = compute_altman_z(df)
    bounds = {}
    for col in RATIO_COLS:
        bounds[col] = (float(df[col].quantile(0.05)), float(df[col].quantile(0.95)))
    del df
    return bounds

def build_features():
    print("Calcul des bornes de winsorisation...")
    bounds = compute_winsor_bounds()

    print("Traitement par année...")
    processed = []
    total_rows = 0
    total_filtered = 0

    # Lit une seule fois, filtre par année
    import duckdb
    for annee in range(2018, 2023):
        conn = duckdb.connect(str(Path("data/riskradar.db")), read_only=True)
        # Lit depuis le parquet via DuckDB — plus efficace
        df = duckdb.read_parquet(str(IN_PATH)).filter(f"annee = {annee}").df()
        conn.close()

        n0 = len(df)
        df = clean_chunk(df)
        df = compute_ratios(df)
        df = compute_altman_z(df)

        for col in RATIO_COLS:
            lo, hi = bounds[col]
            df[col] = df[col].clip(lower=lo, upper=hi)

        total_filtered += n0 - len(df)
        total_rows += len(df)
        processed.append(df)
        print(f"  {annee} : {len(df):,} lignes ({n0-len(df):,} filtrées)")
        del df

    print(f"\nAssemblage final...")
    result = pd.concat(processed, ignore_index=True)
    del processed

    print(f"Total filtrés  : {total_filtered:,}")
    print(f"Total restants : {total_rows:,}")

    print("\nRatios — médianes (sains vs défauts) :")
    print(
        result.groupby('target')[RATIO_COLS]
        .median().round(3)
        .T.rename(columns={0: 'sain', 1: 'défaut'})
    )

    result.to_parquet(OUT_PATH, index=False)
    print(f"\nSauvegardé : {OUT_PATH}")
    
if __name__ == "__main__":
    build_features()
