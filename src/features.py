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

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    # Total actif minimum 1000€
    df = df[df['total_actif'] >= 1000].copy()
    # result_net ne peut pas dépasser 10x total_actif
    mask = df['result_net'].notna() & df['total_actif'].notna()
    df = df[~mask | (df['result_net'].abs() <= df['total_actif'] * 10)]
    # CA ne peut pas dépasser 100x total_actif
    mask2 = df['ca'].notna() & df['total_actif'].notna()
    df = df[~mask2 | (df['ca'] <= df['total_actif'] * 100)]
    print(f"Bilans filtrés : {n0 - len(df):,} supprimés ({(n0-len(df))/n0*100:.1f}%)")
    print(f"Restants       : {len(df):,}")
    return df

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    dettes_totales = df['dettes_lt'].fillna(0) + df['dettes_ct'].fillna(0)

    df['current_ratio']  = safe_div(df['actif_circulant'], df['dettes_ct'])
    df['cash_ratio']     = safe_div(df['tresorerie'],      df['dettes_ct'])
    df['debt_to_equity'] = safe_div(dettes_totales,        df['fonds_propres'])
    df['gearing']        = safe_div(df['dettes_lt'],       df['fonds_propres'])
    df['solvabilite']    = safe_div(df['fonds_propres'],   df['total_actif'])
    df['roe']            = safe_div(df['result_net'],      df['fonds_propres'])
    df['roa']            = safe_div(df['result_net'],      df['total_actif'])
    df['marge_nette']    = safe_div(df['result_net'],      df['ca'])
    df['rotation_actif'] = safe_div(df['ca'],              df['total_actif'])
    return df

def compute_altman_z(df: pd.DataFrame) -> pd.DataFrame:
    ta = df['total_actif'].replace(0, np.nan)
    dt = (df['dettes_lt'].fillna(0) + df['dettes_ct'].fillna(0)).replace(0, np.nan)

    X1 = safe_div(
        df['actif_circulant'].fillna(0) - df['dettes_ct'].fillna(0), ta
    ).clip(-5, 5)
    X4 = safe_div(df['fonds_propres'].fillna(0), dt).clip(-5, 5)
    X5 = safe_div(df['ca'].fillna(0), ta).clip(0, 10)

    # Version simplifiée — sans X2/X3 qui biaisant le score
    df['altman_z'] = (6.56 * X1 + 1.05 * X4 + 1.0 * X5)

    low  = df['altman_z'].quantile(0.05)
    high = df['altman_z'].quantile(0.95)
    df['altman_z'] = df['altman_z'].clip(low, high)

    df['altman_zone'] = pd.cut(
        df['altman_z'],
        bins=[-np.inf, 1.23, 2.90, np.inf],
        labels=['distress', 'grise', 'saine']
    )
    return df
    
def winsorize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        low  = df[col].quantile(0.05)
        high = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=low, upper=high)
    return df

def build_features():
    df = pd.read_parquet(IN_PATH)
    print(f"Dataset chargé : {len(df):,} lignes")

    df = clean_dataset(df)
    df = compute_ratios(df)
    df = compute_altman_z(df)
    df = winsorize(df, RATIO_COLS)

    print("\nRatios — médianes (sains vs défauts) :")
    print(
        df.groupby('target')[RATIO_COLS]
        .median().round(3)
        .T.rename(columns={0: 'sain', 1: 'défaut'})
    )

    print("\nAltman Z-Score — distribution par zone :")
    print(pd.crosstab(
        df['altman_zone'], df['target'],
        rownames=['zone'], colnames=['défaut'],
        normalize='columns'
    ).round(3))

    print("\nAltman Z médiane (sains vs défauts) :")
    print(df.groupby('target')['altman_z'].median().round(3))

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSauvegardé : {OUT_PATH}")

if __name__ == "__main__":
    build_features()
