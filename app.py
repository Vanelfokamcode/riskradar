import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# --- Config ---
st.set_page_config(
    page_title="RiskRadar",
    page_icon="🎯",
    layout="wide"
)

# --- Chargement ---
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_riskradar.json")
    return model

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/features.parquet")
    shap_df = pd.read_parquet("data/processed/shap_values.parquet")
    return df, shap_df

FEATURE_COLS = [
    'current_ratio', 'cash_ratio',
    'debt_to_equity', 'gearing', 'solvabilite',
    'roe', 'roa', 'marge_nette',
    'rotation_actif', 'altman_z',
]

FEATURE_LABELS = {
    'current_ratio':  'Ratio de liquidité',
    'cash_ratio':     'Ratio de trésorerie',
    'debt_to_equity': 'Dette / Fonds propres',
    'gearing':        'Gearing LT',
    'solvabilite':    'Solvabilité',
    'roe':            'ROE',
    'roa':            'ROA',
    'marge_nette':    'Marge nette',
    'rotation_actif': 'Rotation actifs',
    'altman_z':       'Altman Z-Score',
}

def get_zone_color(score: float):
    if score < 0.05:
        return "🟢", "saine", "#2ecc71"
    elif score < 0.20:
        return "🟡", "surveillance", "#f39c12"
    else:
        return "🔴", "distress", "#e74c3c"

def score_row(model, row):
    X = pd.DataFrame(
        [row[FEATURE_COLS].fillna(0).values],
        columns=FEATURE_COLS
    )
    return float(model.predict_proba(X)[:, 1][0])

def plot_gauge(score: float, denomination: str):
    emoji, zone, color = get_zone_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': f"{denomination}<br><sup>Probabilité de défaut (%)</sup>"},
        number={'suffix': "%", 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 5],   'color': "#d5f5e3"},
                {'range': [5, 20],  'color': "#fef9e7"},
                {'range': [20, 100],'color': "#fadbd8"},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=0, l=30, r=30))
    return fig

def plot_history(history_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df['annee'],
        y=history_df['score_pd'] * 100,
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8),
        name='Score PD (%)'
    ))
    fig.add_hrect(y0=0,  y1=5,   fillcolor="#2ecc71", opacity=0.1, line_width=0)
    fig.add_hrect(y0=5,  y1=20,  fillcolor="#f39c12", opacity=0.1, line_width=0)
    fig.add_hrect(y0=20, y1=100, fillcolor="#e74c3c", opacity=0.1, line_width=0)
    fig.update_layout(
        title="Évolution du score de risque",
        xaxis_title="Année",
        yaxis_title="Probabilité de défaut (%)",
        height=350,
        margin=dict(t=50, b=40),
        yaxis=dict(range=[0, 100])
    )
    return fig

def plot_shap_bars(top_features: list):
    labels = [f['label'] for f in top_features]
    values = [f['shap'] for f in top_features]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
    ))
    fig.update_layout(
        title="Top 5 features — impact sur le score (SHAP)",
        xaxis_title="Contribution SHAP",
        height=300,
        margin=dict(t=50, b=40),
    )
    return fig

# --- UI ---
model = load_model()
df, shap_df = load_data()

st.title("🎯 RiskRadar — Credit Risk Scoring")
st.caption("Données réelles INPI × BODACC · XGBoost · SHAP")

# Sidebar
with st.sidebar:
    st.header("🔍 Recherche")
    search_mode = st.radio("Rechercher par", ["SIREN", "Nom d'entreprise"])

    siren_input = None

    if search_mode == "SIREN":
        siren_input = st.text_input("SIREN (9 chiffres)", placeholder="865801401")

    else:
        query = st.text_input("Nom d'entreprise", placeholder="DUPONT")
        if query:
            mask = df['denomination'].str.contains(query, case=False, na=False)
            results = df[mask][['siren','denomination']].drop_duplicates('siren').head(20)
            if not results.empty:
                choice = st.selectbox(
                    "Résultats",
                    results.apply(lambda r: f"{r['siren']} — {r['denomination']}", axis=1)
                )
                siren_input = choice.split(" — ")[0]
            else:
                st.warning("Aucun résultat")

    st.divider()
    st.caption(f"📊 {len(df):,} bilans · {df['siren'].nunique():,} entreprises")

# Main
if not siren_input:
    st.info("👈 Entrez un SIREN ou un nom d'entreprise dans la barre latérale")

    # Stats globales
    st.subheader("📈 Statistiques du dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entreprises", f"{df['siren'].nunique():,}")
    c2.metric("Bilans analysés", f"{len(df):,}")
    c3.metric("Défauts connus", f"{int(df['target'].sum()):,}")
    c4.metric("Taux de défaut", f"{df['target'].mean()*100:.2f}%")

else:
    rows = df[df['siren'] == siren_input]
    if rows.empty:
        st.error(f"SIREN {siren_input} non trouvé dans la base")
    else:
        latest = rows.sort_values('annee').iloc[-1]
        score  = score_row(model, latest)
        emoji, zone, color = get_zone_color(score)
        denomination = str(latest.get('denomination', 'N/A'))

        # Header
        st.subheader(f"{emoji} {denomination}")
        st.caption(f"SIREN : {siren_input} · Dernière année : {int(latest['annee'])}")

        # Métriques top
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score PD", f"{score*100:.2f}%")
        c2.metric("Zone", zone.upper())
        c3.metric("Solvabilité", f"{latest.get('solvabilite', 0):.3f}")
        c4.metric("Gearing", f"{latest.get('gearing', 0):.2f}")

        st.divider()

        # Jauge + Historique
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_gauge(score, denomination), use_container_width=True)

        with col2:
            history_df = rows.sort_values('annee').copy()
            history_df['score_pd'] = history_df.apply(
                lambda r: score_row(model, r), axis=1
            )
            st.plotly_chart(plot_history(history_df), use_container_width=True)

        st.divider()

        # SHAP
        st.subheader("🔎 Pourquoi ce score ?")

        shap_row = shap_df[shap_df['siren'] == siren_input]
        if not shap_row.empty:
            shap_row = shap_row.sort_values('annee').iloc[-1]
            top_features = sorted(
                [
                    {
                        "feature": c,
                        "label":   FEATURE_LABELS[c],
                        "value":   round(float(latest.get(c, 0) or 0), 4),
                        "shap":    round(float(shap_row[f"shap_{c}"]), 4),
                    }
                    for c in FEATURE_COLS
                ],
                key=lambda x: abs(x['shap']),
                reverse=True
            )[:5]

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_shap_bars(top_features), use_container_width=True)
            with col2:
                st.subheader("Détail des features")
                for f in top_features:
                    direction = "🔴 ↑ risque" if f['shap'] > 0 else "🟢 ↓ risque"
                    st.write(f"**{f['label']}** = `{f['value']}` → {direction} (SHAP: `{f['shap']:+.4f}`)")
        else:
            st.info("SHAP values non précalculées pour cette entreprise — elle n'est pas dans le test set 2022")

        st.divider()

        # Ratios financiers
        st.subheader("📊 Ratios financiers")
        # Remplace FEATURE_COLS par :
        DISPLAY_COLS = [c for c in FEATURE_COLS if c != 'altman_z']

        ratio_data = {
            FEATURE_LABELS[c]: round(float(latest.get(c, 0) or 0), 4)
            for c in DISPLAY_COLS
        }
        st.dataframe(
            pd.DataFrame.from_dict(ratio_data, orient='index', columns=['Valeur']),
            use_container_width=True
        )
