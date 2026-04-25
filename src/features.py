# src/features.py
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = Path("data/processed/dataset.parquet")
OUT_PATH = Path("data/processed/features.parquet")

def safe_div(a, b, fill=np.nan):
    """Division sécurisée — évite les div/0 et les infinis."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            (b == 0) | b.isna(),
            fill,
            a / b
        )
    return pd.Series(result, index=a.index)

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule tous les ratios financiers."""

    # --- Liquidité ---
    df['current_ratio'] = safe_div(df['actif_circulant'], df['dettes_ct'])
    df['cash_ratio']    = safe_div(df['tresorerie'],      df['dettes_ct'])

    # --- Endettement ---
    dettes_totales      = df['dettes_lt'].fillna(0) + df['dettes_ct'].fillna(0)
    df['debt_to_equity']= safe_div(dettes_totales,        df['fonds_propres'])
    df['gearing']       = safe_div(df['dettes_lt'],       df['fonds_propres'])
    df['solvabilite']   = safe_div(df['fonds_propres'],   df['total_actif'])

    # --- Rentabilité ---
    df['roe']           = safe_div(df['result_net'],      df['fonds_propres'])
    df['roa']           = safe_div(df['result_net'],      df['total_actif'])
    df['marge_nette']   = safe_div(df['result_net'],      df['ca'])

    # --- Activité ---
    df['rotation_actif']= safe_div(df['ca'],              df['total_actif'])

    return df

# Dans features.py, remplace winsorize par :
def winsorize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        low  = df[col].quantile(0.05)
        high = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=low, upper=high)
    return df

    
def build_features():
    df = pd.read_parquet(IN_PATH)
    print(f"Dataset chargé : {len(df):,} lignes")

    df = compute_ratios(df)
    df = winsorize(df, RATIO_COLS)

    print("\nRatios — médianes (sains vs défauts) :")
    print(
        df.groupby('target')[RATIO_COLS]
        .median()
        .round(3)
        .T
        .rename(columns={0: 'sain', 1: 'défaut'})
    )

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSauvegardé : {OUT_PATH}")

RATIO_COLS = [
    'current_ratio', 'cash_ratio',
    'debt_to_equity', 'gearing', 'solvabilite',
    'roe', 'roa', 'marge_nette',
    'rotation_actif',
]

def build_features():
    df = pd.read_parquet(IN_PATH)
    print(f"Dataset chargé : {len(df):,} lignes")

    df = compute_ratios(df)
    df = winsorize(df, RATIO_COLS)

    # Stats de base pour vérifier
    print("\nRatios — moyennes (sains vs défauts) :")
    print(
        df.groupby('target')[RATIO_COLS]
        .mean()
        .round(3)
        .T
        .rename(columns={0: 'sain', 1: 'défaut'})
    )

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSauvegardé : {OUT_PATH}")

if __name__ == "__main__":
    build_features()