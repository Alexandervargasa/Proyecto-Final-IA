#  EnergyForecast — Predicción de Consumo Eléctrico con ML + LLM

> **Proyecto Final · Inteligencia Artificial · EAFIT 2026-1**

Predicción horaria de demanda energética usando **XGBoost**, **LSTM (PyTorch)** y un componente de **explicabilidad con LLM (LLaMA 3.1 vía Groq)**. Sistema end-to-end reproducible sobre el dataset [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

---

## 🎯 Resultados principales

| Modelo | RMSE (MW) | MAPE (%) | MAE (MW) |
|---|---|---|---|
| Baseline ARIMA | 3,121 | 6.5% | 2,440 |
| XGBoost | 2,034 | 4.2% | 1,589 |
| **LSTM (mejor)** | **1,842** | **3.8%** | **1,401** |

El LSTM supera al baseline en **41% en RMSE** y logra MAPE de **3.8%** (objetivo: < 5%).

---

##  Estructura del repositorio

```
energy-forecast/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb               # EDA + visualizaciones
│   ├── 02_preprocessing.ipynb     # Lag features + secuencias LSTM
│   ├── 03_modeling.ipynb          # ARIMA + XGBoost + LSTM + SHAP
│   └── 04_llm_rag_agents.ipynb    # Explicabilidad con Groq LLM
├── app/
│   └── main.py                    # Demo Streamlit interactiva
├── src/                           # Módulos auxiliares (opcional)
├── data/
│   ├── raw/                       # Dataset original (no en git)
│   └── processed/                 # Datos procesados + figuras
├── models/
│   └── checkpoints/               # Pesos guardados (no en git)
└── docs/
    └── informe_final.pdf          # Informe LaTeX compilado
```

---

##  Instalación y ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/equipo-ia-eafit/energy-forecast-2026.git
cd energy-forecast-2026
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset
1. Ir a https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
2. Descargar `PJME_hourly.csv`
3. Colocarlo en `data/raw/PJME_hourly.csv`

### 4. Ejecutar notebooks en orden
```bash
# En Jupyter / Colab:
01_eda.ipynb
02_preprocessing.ipynb
03_modeling.ipynb
04_llm_rag_agents.ipynb   # requiere GROQ_API_KEY
```

### 5. Demo Streamlit (opcional)
```bash
export GROQ_API_KEY=gsk_tu_key_aqui
streamlit run app/main.py
```

---

##  API Key de Groq (gratuita)

1. Regístrate en https://console.groq.com
2. Ve a **API Keys → Create Key**
3. Úsala en el notebook 04 o como variable de entorno:
   ```bash
   export GROQ_API_KEY=gsk_...
   ```

---

##  Stack tecnológico

| Componente | Tecnología |
|---|---|
| EDA | pandas, matplotlib, seaborn, statsmodels |
| Feature Engineering | Lag features, rolling stats, encodings cíclicos |
| Modelo ML | XGBoost 2.0 |
| Modelo DL | LSTM 2×64 (PyTorch 2.3) |
| Explicabilidad ML | SHAP TreeExplainer |
| LLM | LLaMA 3.1 8B via Groq API |
| Estrategia prompting | Few-shot + chain-of-thought |
| Demo | Streamlit 1.35 |
| Hardware | Google Colab (GPU T4) |

---

##  Equipo

| Integrante | Correo | Contribución |
|---|---|---|
| Sofia Velez | Svelezr10@eafit.edu.co | EDA, feature engineering, visualizaciones |
| Paulina Velasquez | pvelasquel@eafit.edu.co | Arquitectura LSTM, entrenamiento, evaluación |
| Alexander Vargas | avargas@eafit.edu.co | LLM + SHAP, integración, informe |

---

## 🎥 Video demo

📹 **[Ver demo en YouTube](https://youtu.be/LINK-AL-VIDEO)** (≤ 3 minutos)

---

## 📄 Informe

El informe completo en PDF está en [`docs/informe_final.pdf`](docs/informe_final.pdf), generado con la plantilla LaTeX de EAFIT en Overleaf.

---

## 📚 Referencias principales

- Hong & Fan (2016). Probabilistic electric load forecasting. *IJF*.
- Chen & Guestrin (2016). XGBoost. *KDD 2016*.
- Hochreiter & Schmidhuber (1997). LSTM. *Neural Computation*.
- Lundberg & Lee (2017). SHAP. *NeurIPS 2017*.
- Dataset: Rob Mulla (2018). PJM Hourly Energy Consumption. Kaggle.
