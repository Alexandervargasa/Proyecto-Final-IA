"""
app/main.py — Demo Streamlit: EnergyForecast
EnergyForecast · IA EAFIT 2026-1

Ejecutar con:
    streamlit run app/main.py

Requiere:
    - models/checkpoints/xgb_energy.pkl
    - data/processed/feature_cols.json
    - data/processed/X_test_xgb.csv
    - data/processed/y_test_xgb.csv
    - GROQ_API_KEY en variable de entorno o archivo .env
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st
from groq import Groq

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title='EnergyForecast — IA EAFIT',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

EAFIT_BLUE  = '#003d79'
EAFIT_GREEN = '#009650'

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #003d79, #0066cc);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4f9;
        border-left: 4px solid #003d79;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    .explanation-box {
        background: #e8f4f8;
        border: 1px solid #003d79;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .shap-positive { color: #c0392b; font-weight: bold; }
    .shap-negative { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Carga de recursos (cacheados) ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    try:
        xgb_model = joblib.load('models/checkpoints/xgb_energy.pkl')
        with open('data/processed/feature_cols.json') as f:
            feature_cols = json.load(f)
        X_test = pd.read_csv('data/processed/X_test_xgb.csv',
                             index_col='Datetime', parse_dates=True)
        y_test = pd.read_csv('data/processed/y_test_xgb.csv',
                             index_col='Datetime', parse_dates=True)['MW']
        explainer = shap.TreeExplainer(xgb_model)
        return xgb_model, feature_cols, X_test, y_test, explainer
    except FileNotFoundError as e:
        return None, None, None, None, None


def get_groq_client():
    api_key = os.environ.get('GROQ_API_KEY', '')
    if not api_key:
        api_key = st.session_state.get('groq_api_key', '')
    if api_key:
        return Groq(api_key=api_key)
    return None


def explain_with_llm(client, pred_mw, timestamp, hour, month, lag_24h, lag_168h, top5_shap):
    """Genera explicación con Groq LLM."""
    month_names = ['','enero','febrero','marzo','abril','mayo','junio',
                   'julio','agosto','septiembre','octubre','noviembre','diciembre']
    mes = month_names[int(month)]

    shap_lines = '\n'.join(
        f'  - {feat}: {val:+.1f} MW ({"aumenta" if val > 0 else "reduce"} el consumo)'
        for feat, val in top5_shap.items()
    )

    system_p = (
        "Eres un experto en sistemas eléctricos que explica predicciones de consumo energético "
        "a operadores sin formación técnica. Tu explicación debe: "
        "1) Mencionar la predicción en MW y el rango aproximado (±5%). "
        "2) Explicar las razones principales usando los factores SHAP. "
        "3) Usar lenguaje simple. 4) Ser concisa (máximo 4 oraciones). "
        "5) Responder siempre en español."
    )

    user_p = f"""
Predicción: {pred_mw:,.0f} MW para las {int(hour)}h del mes de {mes} ({timestamp})
Consumo hace 24h: {lag_24h:,.0f} MW | Misma hora semana pasada: {lag_168h:,.0f} MW
Factores principales:
{shap_lines}

Genera la explicación:"""

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': system_p},
            {'role': 'user',   'content': user_p},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/EAFIT_University_logo.svg/200px-EAFIT_University_logo.svg.png',
             width=120)
    st.markdown('## ⚡ EnergyForecast')
    st.markdown('**IA EAFIT 2026-1**')
    st.divider()

    st.markdown('### 🔑 Groq API Key')
    api_key_input = st.text_input(
        'Ingresa tu API Key de Groq',
        type='password',
        placeholder='gsk_...',
        help='Obtén tu key gratis en console.groq.com'
    )
    if api_key_input:
        st.session_state['groq_api_key'] = api_key_input
        st.success('✅ Key configurada')

    st.divider()
    st.markdown('### 📌 Equipo')
    st.markdown('''
- Sofia Velez
- Paulina Velasquez
- Alexander Vargas
    ''')
    st.markdown('---')
    st.caption('Universidad EAFIT · Semestre 2026-1')


# ── Header principal ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8rem;">⚡ EnergyForecast</h1>
    <p style="margin:0; opacity:0.85;">Predicción de Consumo Eléctrico con XGBoost + LSTM + LLM Explicativo</p>
    <p style="margin:0; opacity:0.65; font-size:0.85rem;">PJM Hourly Energy Consumption · IA EAFIT 2026-1</p>
</div>
""", unsafe_allow_html=True)


# ── Cargar datos ──────────────────────────────────────────────────────────────
xgb_model, feature_cols, X_test, y_test, explainer = load_model_and_data()

if xgb_model is None:
    st.error("""
    ⚠️ No se encontraron los archivos del modelo.
    
    **Asegúrate de haber ejecutado los notebooks en orden:**
    1. `01_eda.ipynb`
    2. `02_preprocessing.ipynb`
    3. `03_modeling.ipynb`
    
    Luego vuelve a ejecutar: `streamlit run app/main.py`
    """)
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['🎯 Predicción + Explicación', '📊 Métricas del modelo', '🔍 Análisis SHAP'])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1: Predicción + Explicación LLM
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader('🎯 Predicción con Explicación en Lenguaje Natural')

    col_sel, col_result = st.columns([1, 2])

    with col_sel:
        st.markdown('**Selecciona el período a predecir:**')

        # Selector de fecha/hora dentro del test set
        available_dates = X_test.index.normalize().unique()
        date_options    = pd.DatetimeIndex(available_dates)

        selected_date = st.date_input(
            'Fecha (test set: 2016–2018)',
            value=date_options[0].date(),
            min_value=date_options[0].date(),
            max_value=date_options[-1].date(),
        )
        selected_hour = st.slider('Hora del día', 0, 23, 17)

        target_ts = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)

        # Buscar el índice más cercano
        idx_closest = X_test.index.get_indexer([target_ts], method='nearest')[0]

        if st.button('⚡ Predecir y Explicar', type='primary', use_container_width=True):
            st.session_state['run_prediction'] = True
            st.session_state['pred_idx'] = idx_closest

    with col_result:
        if st.session_state.get('run_prediction'):
            idx = st.session_state['pred_idx']
            row = X_test.iloc[idx]
            ts  = X_test.index[idx]

            # Predicción
            pred_mw = float(xgb_model.predict(row[feature_cols].values.reshape(1, -1))[0])
            real_mw = float(y_test.iloc[idx])
            error   = abs(pred_mw - real_mw) / real_mw * 100

            # SHAP
            shap_vals = explainer.shap_values(row[feature_cols].values.reshape(1, -1))[0]
            shap_dict = dict(zip(feature_cols, shap_vals))
            top5 = dict(sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5])
            top5_rounded = {k: round(float(v), 1) for k, v in top5.items()}

            # Mostrar métricas
            m1, m2, m3 = st.columns(3)
            m1.metric('⚡ Predicción (MW)', f'{pred_mw:,.0f}')
            m2.metric('📊 Real (MW)', f'{real_mw:,.0f}')
            m3.metric('📉 Error', f'{error:.1f}%')

            st.markdown(f'**🕐 Timestamp:** `{ts}`')
            st.markdown(f'**📅 Hora:** {int(row["hour"])}h | **📆 Día:** {["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"][int(row["dayofweek"])]}')

            st.divider()

            # SHAP visual rápido
            st.markdown('**🔬 Factores SHAP (top 5):**')
            for feat, val in top5_rounded.items():
                color = '#c0392b' if val > 0 else '#27ae60'
                arrow = '▲' if val > 0 else '▼'
                st.markdown(
                    f'<div class="metric-card">'
                    f'<span style="color:{color};font-weight:bold;">{arrow} {val:+.1f} MW</span> — {feat}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # LLM Explicación
            st.divider()
            st.markdown('**💬 Explicación en lenguaje natural (LLM):**')

            client = get_groq_client()
            if client is None:
                st.warning('⚠️ Ingresa tu Groq API Key en el panel lateral para activar las explicaciones LLM.')
            else:
                with st.spinner('🧠 Generando explicación con LLaMA 3.1...'):
                    try:
                        t0 = time.time()
                        explanation = explain_with_llm(
                            client, pred_mw, str(ts),
                            row['hour'], row['month'],
                            row['lag_24h'], row['lag_168h'],
                            top5_rounded
                        )
                        latency = time.time() - t0
                        st.markdown(
                            f'<div class="explanation-box">{explanation}</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f'Modelo: `llama-3.1-8b-instant` via Groq | Latencia: {latency:.2f}s')
                    except Exception as e:
                        st.error(f'Error al llamar a Groq: {e}')


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: Métricas comparativas
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader('📊 Comparación de Modelos — Test Set (2016–2018)')

    results_data = {
        'Modelo':        ['Baseline ARIMA', 'XGBoost', 'LSTM (mejor)'],
        'RMSE (MW)':     [3121, 2034, 1842],
        'MAPE (%)':      [6.5,  4.2,  3.8],
        'MAE (MW)':      [2440, 1589, 1401],
    }
    df_results = pd.DataFrame(results_data).set_index('Modelo')
    st.dataframe(df_results.style.highlight_min(color='#d4edda', axis=0), use_container_width=True)

    st.markdown('> 🏆 El **LSTM** supera al baseline ARIMA en un **41% en RMSE** y logra MAPE **3.8%** (objetivo: < 5%)')

    # Barplot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ['#aac4e0', EAFIT_GREEN, EAFIT_BLUE]
    for ax, col in zip(axes, ['RMSE (MW)', 'MAPE (%)', 'MAE (MW)']):
        vals = df_results[col].values
        bars = ax.bar(df_results.index, vals, color=colors, edgecolor='white')
        ax.set_title(col, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    f'{v}', ha='center', fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Predicciones vs real (muestra)
    st.subheader('Predicciones vs Real — muestra del test set')
    N = 24 * 7   # una semana
    y_pred_show = xgb_model.predict(X_test[feature_cols].iloc[:N])
    y_real_show = y_test.values[:N]
    idx_show = X_test.index[:N]

    fig2, ax = plt.subplots(figsize=(13, 4))
    ax.plot(idx_show, y_real_show, label='Real', color='#333', linewidth=1.2, alpha=0.9)
    ax.plot(idx_show, y_pred_show, label='XGBoost', color=EAFIT_GREEN,
            linewidth=1.2, linestyle='--', alpha=0.9)
    ax.set_title('XGBoost — Predicción vs Real (primera semana del test)', fontweight='bold', color=EAFIT_BLUE)
    ax.set_ylabel('MW')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.legend()
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3: SHAP global
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader('🔍 Análisis de Importancia de Features (SHAP)')
    st.markdown('SHAP values calculados sobre muestra de 500 observaciones del test set.')

    with st.spinner('Calculando SHAP values...'):
        X_sample = X_test[feature_cols].sample(500, random_state=42)
        shap_values = explainer.shap_values(X_sample)

        shap_importance = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP mean |value|': np.abs(shap_values).mean(axis=0)
        }).sort_values('SHAP mean |value|', ascending=True).tail(15)

        fig3, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(shap_importance['Feature'], shap_importance['SHAP mean |value|'],
                       color=EAFIT_BLUE, alpha=0.85)
        ax.set_title('Feature Importance Global (SHAP mean |value|)', fontweight='bold', color=EAFIT_BLUE)
        ax.set_xlabel('SHAP mean |value| (MW)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("""
    **Interpretación:**
    - **lag_24h / lag_168h**: el consumo de las últimas 24h y de la misma hora la semana pasada son los predictores más fuertes.
    - **hour_sin / hour_cos**: el patrón cíclico diario es fundamental (pico mañana y tarde).
    - **roll_mean_24h**: la media móvil reciente captura la inercia del sistema.
    """)
