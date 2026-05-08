"""Dashboard Interactivo de Modelos ARIMA - Versión Educativa Mejorada"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from datetime import datetime, timedelta
from scipy import stats
import seaborn as sns
warnings.filterwarnings("ignore")

# Configuración de página
st.set_page_config(
    page_title="Dashboard ARIMA - Versión Educativa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados
st.markdown("""
<style>
    /* Fondo principal */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Títulos */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(0, 219, 222, 0.3);
    }
    
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00DBDE;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #00DBDE, #FC00FF) 1;
    }
    
    .subsection-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FC00FF;
        margin: 2rem 0 1rem 0;
    }
    
    /* Cajas de información */
    .tech-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-left: 5px solid #00DBDE;
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 219, 222, 0.2);
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(252,0,255,0.1) 0%, rgba(252,0,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(252,0,255,0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(252,0,255,0.2);
    }
    
    .formula-box {
        background: linear-gradient(135deg, rgba(0,219,222,0.15) 0%, rgba(0,219,222,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid #00DBDE;
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 219, 222, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255,107,107,0.15) 0%, rgba(255,107,107,0.05) 100%);
        backdrop-filter: blur(10px);
        border-left: 5px solid #FF6B6B;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(78,205,196,0.15) 0%, rgba(78,205,196,0.05) 100%);
        backdrop-filter: blur(10px);
        border-left: 5px solid #4ECDC4;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
    }
    
    /* Métricas */
    .metric-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 219, 222, 0.3);
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 219, 222, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 219, 222, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 219, 222, 0.6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(15,12,41,0.95) 0%, rgba(48,43,99,0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* Texto general */
    .stMarkdown {
        color: #e0e0e0;
        line-height: 1.8;
    }
    
    /* Select boxes */
    .stSelectbox>div>div, .stMultiselect>div>div {
        background: rgba(255,255,255,0.1);
        border: 2px solid rgba(0, 219, 222, 0.3);
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(0,219,222,0.1) 0%, rgba(252,0,255,0.1) 100%);
        border-radius: 10px;
        border: 2px solid rgba(0, 219, 222, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar AirPassengers (necesaria para ejemplos)
def load_airpassengers():
    """Carga el dataset clásico AirPassengers (pasajeros aéreos mensuales 1949-1960)"""
    from statsmodels.datasets import get_rdataset
    try:
        data = get_rdataset("AirPassengers").data
        data['time'] = pd.date_range(start='1949-01-01', periods=len(data), freq='MS')
        data = data.set_index('time')
        return data['value']
    except:
        # Datos hardcodeados por si falla la descarga
        dates = pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS')
        values = [112,118,132,129,121,135,148,148,136,119,104,118,
                  115,126,141,135,125,149,170,170,158,133,114,140,
                  145,150,178,163,172,178,199,199,184,162,146,166,
                  171,180,193,181,183,218,230,242,209,191,172,194,
                  196,196,236,235,229,243,264,272,237,211,180,201,
                  204,188,235,227,234,264,302,293,259,229,203,229,
                  242,233,267,269,270,315,364,347,312,274,237,278,
                  284,277,317,313,318,374,413,405,355,306,271,306,
                  315,301,356,348,355,422,465,467,404,347,305,336,
                  340,318,362,348,363,435,491,505,404,359,310,337,
                  360,342,406,396,420,472,548,559,463,407,362,405,
                  417,391,419,461,472,535,622,606,508,461,390,432]
        return pd.Series(values, index=dates, name='Pasajeros (miles)')

# Funciones para cargar datasets econométricos
def load_gnp():
    """Carga el Producto Nacional Bruto (GNP) de EE.UU. - Dataset clásico de econometría"""
    from statsmodels.datasets import get_rdataset
    try:
        data = get_rdataset("gnp").data
        data = data.set_index('date')
        return data['value']
    except:
        # Datos simulados basados en patrones reales del GNP
        dates = pd.date_range(start='1947-01-01', end='2023-12-01', freq='QE')
        np.random.seed(42)
        n = len(dates)
        trend = 300 + np.arange(n) * 3.5 + 0.01 * np.arange(n)**1.5
        cycle = 15 * np.sin(2 * np.pi * np.arange(n) / 20)  # Ciclo económico ~5 años
        noise = np.random.normal(0, 5, n)
        values = trend + cycle + noise
        return pd.Series(values, index=dates, name='GNP (miles de millones USD)')

def load_unemployment():
    """Carga la tasa de desempleo de EE.UU. - Serie económica fundamental"""
    from statsmodels.datasets import get_rdataset
    try:
        data = get_rdataset("unemployment").data
        data = data.set_index('date')
        return data['value']
    except:
        # Datos simulados basados en patrones reales de desempleo
        dates = pd.date_range(start='1948-01-01', end='2023-12-01', freq='MS')
        np.random.seed(42)
        n = len(dates)
        base = 5.5
        cycle = 2.5 * np.sin(2 * np.pi * np.arange(n) / 72)  # Ciclo ~6 años
        crisis = np.zeros(n)
        # Crisis económicas simuladas
        crisis[240:252] = 4  # Crisis 2008
        crisis[876:900] = 8  # Crisis COVID 2020
        noise = np.random.normal(0, 0.3, n)
        values = base + cycle + crisis + noise
        values = np.clip(values, 2.5, 15)
        return pd.Series(values, index=dates, name='Tasa de Desempleo (%)')

def load_inflation():
    """Carga el Índice de Precios al Consumidor (IPC) - Inflación"""
    from statsmodels.datasets import get_rdataset
    try:
        data = get_rdataset("inf")  # Inflation data
        data = data.set_index('date')
        return data['value']
    except:
        # Datos simulados basados en patrones reales de inflación
        dates = pd.date_range(start='1960-01-01', end='2023-12-01', freq='MS')
        np.random.seed(42)
        n = len(dates)
        base = 3.0
        # Regímenes de inflación
        high_inf = ((dates.year >= 1973) & (dates.year <= 1982)).astype(float)
        low_inf = (dates.year >= 1990).astype(float)
        regime = 2 * high_inf - 1.5 * low_inf
        persistence = np.zeros(n)
        for i in range(1, n):
            persistence[i] = 0.7 * persistence[i-1] + np.random.normal(0, 0.5)
        values = base + regime + persistence
        values = np.clip(values, -2, 15)
        return pd.Series(values, index=dates, name='Inflación IPC (%)')

def load_interest_rate():
    """Carga la tasa de interés de los Fed Funds - Política monetaria"""
    from statsmodels.datasets import get_rdataset
    try:
        data = get_rdataset("interest")  # Interest rate data
        data = data.set_index('date')
        return data['value']
    except:
        # Datos simulados basados en patrones reales de tasas de interés
        dates = pd.date_range(start='1955-01-01', end='2023-12-01', freq='MS')
        np.random.seed(42)
        n = len(dates)
        # Regímenes de política monetaria
        base = 4.0
        high_rate = ((dates.year >= 1979) & (dates.year <= 1985)).astype(float)
        low_rate = (dates.year >= 2008).astype(float)
        regime = 6 * high_rate - 3.5 * low_rate
        cycle = 1.5 * np.sin(2 * np.pi * np.arange(n) / 96)  # Ciclo ~8 años
        noise = np.random.normal(0, 0.3, n)
        values = base + regime + cycle + noise
        values = np.clip(values, 0, 20)
        return pd.Series(values, index=dates, name='Tasa de Interés (%)')

def load_gdp_growth():
    """Carga el crecimiento del PIB - Indicador económico clave"""
    from statsmodels.datasets import get_rdataset
    try:
        data = get_rdataset("gdp_growth")
        data = data.set_index('date')
        return data['value']
    except:
        # Datos simulados basados en patrones reales de crecimiento del PIB
        dates = pd.date_range(start='1947-01-01', end='2023-12-01', freq='QE')
        np.random.seed(42)
        n = len(dates)
        base = 3.0
        cycle = 2 * np.sin(2 * np.pi * np.arange(n) / 24)  # Ciclo ~6 años
        recession = np.zeros(n)
        # Recesiones simuladas
        recession[40:44] = -8  # Recesión 1957
        recession[120:126] = -10  # Recesión 1973
        recession[200:204] = -8  # Recesión 1990
        recession[240:248] = -12  # Crisis 2008
        recession[290:294] = -15  # Crisis COVID 2020
        noise = np.random.normal(0, 1, n)
        values = base + cycle + recession + noise
        return pd.Series(values, index=dates, name='Crecimiento PIB (%)')

# Diccionario de datasets econométricos disponibles
DATASETS = {
    "📈 GNP - Producto Nacional Bruto (1947-2023)": {
        "func": load_gnp,
        "desc": "Producto Nacional Bruto de EE.UU. en miles de millones de dólares. Serie clásica de econometría con tendencia creciente y ciclos económicos.",
        "freq": "Trimestral",
        "period": "1947-2023",
        "ideal_for": "ARIMA con tendencia (d=1 o d=2)"
    },
    "👥 Tasa de Desempleo (1948-2023)": {
        "func": load_unemployment,
        "desc": "Tasa de desempleo mensual de EE.UU. Muestra ciclos económicos, crisis (2008, 2020) y mean reversion.",
        "freq": "Mensual",
        "period": "1948-2023",
        "ideal_for": "ARMA o ARIMA con d=0 (estacionaria)"
    },
    "💰 Inflación IPC (1960-2023)": {
        "func": load_inflation,
        "desc": "Índice de Precios al Consumidor (inflación mensual). Muestra regímenes de alta y baja inflación con persistencia.",
        "freq": "Mensual",
        "period": "1960-2023",
        "ideal_for": "ARIMA con cambios de régimen"
    },
    "🏦 Tasa de Interés Fed Funds (1955-2023)": {
        "func": load_interest_rate,
        "desc": "Tasa de interés de los Fondos Federales. Refleja política monetaria con regímenes de alta y baja tasa.",
        "freq": "Mensual",
        "period": "1955-2023",
        "ideal_for": "ARIMA con cambios estructurales"
    },
    "📊 Crecimiento del PIB (1947-2023)": {
        "func": load_gdp_growth,
        "desc": "Crecimiento trimestral del PIB real. Muestra ciclos económicos, recesiones y recuperación.",
        "freq": "Trimestral",
        "period": "1947-2023",
        "ideal_for": "ARMA o ARIMA con d=0 (estacionaria)"
    }
}

# Funciones auxiliares
def test_stationarity(series):
    result = adfuller(series.dropna(), autolag='AIC')
    output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': {
            '1%': result[4]['1%'],
            '5%': result[4]['5%'],
            '10%': result[4]['10%']
        }
    }
    return output, result[1] <= 0.05

def fit_arima_model(data, order):
    try:
        model = ARIMA(data, order=order)
        return model.fit()
    except:
        return None

def evaluate_model(model_fit, test_data):
    try:
        forecast = model_fit.forecast(steps=len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        mae = mean_absolute_error(test_data, forecast)
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'forecast': forecast
        }
    except:
        return None

# Funciones de visualización mejoradas
def plot_time_series_advanced(data, title, dataset_info):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(
            f"{title} - Serie Temporal",
            "Distribución",
            "Descomposición (Tendencia)",
            "Estacionalidad"
        )
    )
    
    # Serie temporal principal
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data.values,
            mode='lines',
            name='Serie',
            line=dict(
                color='#00DBDE',
                width=2.5
            ),
            fill='tozeroy',
            fillcolor='rgba(0, 219, 222, 0.1)'
        ),
        row=1, col=1
    )
    
    # Distribución
    fig.add_trace(
        go.Histogram(
            x=data.values,
            name='Distribución',
            nbinsx=30,
            marker_color='#FC00FF',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Tendencia (media móvil)
    if len(data) > 12:
        trend = data.rolling(window=12).mean()
        fig.add_trace(
            go.Scatter(
                x=trend.index, y=trend.values,
                mode='lines',
                name='Tendencia (MA12)',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ),
            row=3, col=1
        )
    
    # Estacionalidad (si es mensual)
    if hasattr(data.index, 'month') and len(data) >= 24:
        monthly_avg = data.groupby(data.index.month).mean()
        fig.add_trace(
            go.Bar(
                x=monthly_avg.index,
                y=monthly_avg.values,
                name='Promedio Mensual',
                marker_color='#4ECDC4',
                opacity=0.7
            ),
            row=4, col=1
        )
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=12)
    )
    
    return fig

def plot_acf_pacf_advanced(series, lags=40):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "ACF - Autocorrelación (Identifica q - MA)",
            "PACF - Autocorrelación Parcial (Identifica p - AR)"
        )
    )
    
    try:
        acf_values = acf(series, nlags=lags, fft=True)
        colors_acf = ['#00DBDE' if v >= 0 else '#FC00FF' for v in acf_values]
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF',
                marker_color=colors_acf,
                opacity=0.8
            ),
            row=1, col=1
        )
        sig = 1.96 / np.sqrt(len(series))
        fig.add_hline(y=sig, line_dash="dash", line_color="#FF6B6B", row=1, col=1)
        fig.add_hline(y=-sig, line_dash="dash", line_color="#FF6B6B", row=1, col=1)
    except:
        pass
    
    try:
        pacf_values = pacf(series, nlags=lags, method='yw')
        colors_pacf = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in pacf_values]
        fig.add_trace(
            go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                name='PACF',
                marker_color=colors_pacf,
                opacity=0.8
            ),
            row=2, col=1
        )
        fig.add_hline(y=sig, line_dash="dash", line_color="#FF6B6B", row=2, col=1)
        fig.add_hline(y=-sig, line_dash="dash", line_color="#FF6B6B", row=2, col=1)
    except:
        pass
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=12)
    )
    
    return fig

def plot_forecast_advanced(train, test, forecast, model_fit, title, order):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.25, 0.2],
        subplot_titles=(
            f"Pronóstico ARIMA{order} - {title}",
            "Residuos del Modelo",
            "Q-Q Plot (Normalidad)"
        )
    )
    
    # Serie de entrenamiento
    fig.add_trace(
        go.Scatter(
            x=train.index, y=train.values,
            mode='lines',
            name='Entrenamiento',
            line=dict(color='#00DBDE', width=2)
        ),
        row=1, col=1
    )
    
    # Serie de prueba (valores reales)
    if test is not None and len(test) > 0:
        fig.add_trace(
            go.Scatter(
                x=test.index, y=test.values,
                mode='lines',
                name='Valor Real (Test)',
                line=dict(color='#4ECDC4', width=2)
            ),
            row=1, col=1
        )
    
    # Pronóstico
    forecast_index = pd.date_range(
        start=train.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast),
        freq='D' if train.index.freq == 'D' else 'MS'
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_index, y=forecast,
            mode='lines+markers',
            name='Pronóstico ARIMA',
            line=dict(color='#FC00FF', width=2, dash='dash'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Intervalo de confianza
    try:
        conf_int = model_fit.get_forecast(steps=len(forecast)).conf_int()
        fig.add_trace(
            go.Scatter(
                x=forecast_index, y=conf_int.iloc[:, 1],
                mode='lines',
                name='Límite Superior (95%)',
                line=dict(color='rgba(252,0,255,0.3)', width=1),
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_index, y=conf_int.iloc[:, 0],
                mode='lines',
                name='Límite Inferior (95%)',
                fill='tonexty',
                fillcolor='rgba(0,219,222,0.1)',
                line=dict(color='rgba(252,0,255,0.3)', width=1),
                showlegend=True
            ),
            row=1, col=1
        )
    except:
        pass
    
    # Residuos
    residuals = model_fit.resid
    colors_resid = ['#4ECDC4' if r >= 0 else '#FF6B6B' for r in residuals]
    fig.add_trace(
        go.Bar(
            x=residuals.index, y=residuals.values,
            name='Residuos',
            marker_color=colors_resid,
            opacity=0.7
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_color="#888888", line_width=1, row=2, col=1)
    
    # Q-Q Plot
    try:
        qq = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq[0][0], y=qq[0][1],
                mode='markers',
                name='Residuos',
                marker=dict(color='#00DBDE', size=6)
            ),
            row=3, col=1
        )
        min_val, max_val = min(qq[0][0]), max(qq[0][0])
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Referencia Normal',
                line=dict(color='#FF6B6B', dash='dash')
            ),
            row=3, col=1
        )
    except:
        pass
    
    fig.update_layout(
        height=900,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=12)
    )
    
    return fig

# Función principal
def main():
    # Header principal
    st.markdown('<h1 class="main-header">📊 Dashboard Interactivo de Modelos ARIMA</h1>', unsafe_allow_html=True)
    
    # Descripción initial
    st.markdown("""
    <div class="tech-box">
        <h3 style="color: #00DBDE; margin-top: 0;">🎯 Objetivo del Dashboard</h3>
        <p style="color: #e0e0e0; line-height: 1.8; font-size: 1.1rem;">
            Esta herramienta educativa te permitirá comprender y aplicar modelos ARIMA utilizando 
            <strong style="color: #4ECDC4;">datasets clásicos y educativos</strong> que muestran patrones claros. 
            Explora diferentes comportamientos de series temporales (tendencia, estacionalidad, ciclos) 
            y aprende a identificar el modelo ARIMA adecuado para cada caso.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar de navegación
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0,219,222,0.2) 0%, rgba(252,0,255,0.2) 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h3 style="color: #00DBDE; margin-top: 0; font-size: 1.5rem;">🧭 Navegación</h3>
        <p style="color: #e0e0e0; font-size: 0.9rem;">Selecciona una sección para comenzar</p>
    </div>
    """, unsafe_allow_html=True)
    
    opcion = st.sidebar.radio(
        "Selecciona una sección:",
        [
            "📚 Fundamentos Teóricos de ARIMA",
            "🔍 Análisis Exploratorio con Datasets Educativos",
            "🎯 5 Ejemplos ARIMA Paso a Paso",
            "📊 Comparativa de Modelos",
            "🧪 Laboratorio Experimental",
            "🔄 AR vs MA vs ARMA vs ARIMA vs SARIMA",
            "⚠️ Limitaciones y Casos de Uso"
        ],
        label_visibility="collapsed"
    )
    
    # ==================== SECCIÓN 1: FUNDAMENTOS TEÓRICOS ====================
    if opcion == "📚 Fundamentos Teóricos de ARIMA":
        st.markdown('<h2 class="section-header">📚 Fundamentos Teóricos de ARIMA</h2>', unsafe_allow_html=True)
        
        # Introducción completa a ARIMA
        st.markdown("""
        <div class="tech-box">
            <h4 style="color: #00DBDE;">📖 ¿Qué es ARIMA y por qué es fundamental en Econometría?</h4>
            <p style="color: #e0e0e0; line-height: 1.8; font-size: 1.05rem;">
                <strong style="color: #4ECDC4;">ARIMA</strong> (AutoRegressive Integrated Moving Average) es uno de los modelos 
                <strong style="color: #FC00FF;">más importantes y utilizados en econometría</strong> para el análisis y pronóstico 
                de series temporales. Desarrollado por <strong style="color: #00DBDE;">Box y Jenkins</strong> en 1970, 
                ARIMA se ha convertido en el <strong style="color: #4ECDC4;">estándar de oro</strong> para modelar 
                series temporales univariadas en economía, finanzas, negocios y ciencias sociales.
            </p>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(0,219,222,0.1); border-radius: 10px; border-left: 4px solid #00DBDE;">
                <p style="color: #e0e0e0; line-height: 1.8; margin: 0;">
                    <strong style="color: #FC00FF;">¿Por qué ARIMA es la mejor opción?</strong><br>
                    • <strong style="color: #00DBDE;">Flexibilidad:</strong> Captura una amplia gama de patrones (tendencia, estacionalidad, ciclos)<br>
                    • <strong style="color: #4ECDC4;">Rigor estadístico:</strong> Basado en teoría sólida con pruebas de diagnóstico<br>
                    • <strong style="color: #FF6B6B;">Interpretabilidad:</strong> Cada parámetro tiene significado económico claro<br>
                    • <strong style="color: #FFD93D;">Versatilidad:</strong> Aplicable a múltiples campos (economía, finanzas, meteorología)<br>
                    • <strong style="color: #4ECDC4;">Eficiencia:</strong> Mínimos cuadrados y máxima verosimilitud garantizan optimalidad
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Importancia en econometría
        st.markdown("""
        <div class="info-card" style="margin-top: 2rem;">
            <h4 style="color: #4ECDC4;">🏛️ Importancia de ARIMA en Econometría</h4>
            <p style="color: #e0e0e0; line-height: 1.8;">
                En econometría, ARIMA es <strong style="color: #00DBDE;">fundamental</strong> porque:
            </p>
            <ul style="color: #e0e0e0; line-height: 2; font-size: 1.05rem;">
                <li>📊 <strong style="color: #4ECDC4;">Modela la dinámica temporal:</strong> Captura cómo el pasado influye en el futuro, esencial para políticas económicas</li>
                <li>🔮 <strong style="color: #00DBDE;">Pronósticos confiables:</strong> Base para proyecciones de PIB, inflación, desempleo, ventas</li>
                <li>📈 <strong style="color: #FF6B6B;">Análisis de impactos:</strong> Permite evaluar efectos de shocks económicos (crisis, políticas)</li>
                <li>🎯 <strong style="color: #FFD93D;">Toma de decisiones:</strong> Empresas y gobiernos usan ARIMA para planificación estratégica</li>
                <li>🔬 <strong style="color: #4ECDC4;">Base para modelos avanzados:</strong> VAR, VECM, GARCH se construyen sobre fundamentos ARIMA</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # ¿Qué es ARIMA? - Explicación detallada
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #4ECDC4;">¿Qué es ARIMA?</h4>
                <p style="color: #e0e0e0; line-height: 1.8;">
                    <strong style="color: #00DBDE;">ARIMA</strong> (AutoRegressive Integrated Moving Average) 
                    es un modelo estadístico para analizar y pronosticar series temporales. Combina tres componentes:
                </p>
                <ul style="color: #e0e0e0; line-height: 2;">
                    <li><strong style="color: #00DBDE;">AR (Autoregresivo - p):</strong> 
                        El valor actual depende de sus propios valores pasados. 
                        <em style="color: #a0a0a0;">"El pasado reciente influye en el presente"</em></li>
                    <li><strong style="color: #4ECDC4;">I (Integrado - d):</strong> 
                        Número de diferenciaciones necesarias para hacer la serie estacionaria. 
                        <em style="color: #a0a0a0;">"Eliminamos tendencias"</em></li>
                    <li><strong style="color: #FC00FF;">MA (Media Móvil - q):</strong> 
                        El valor actual depende de errores de pronósticos pasados. 
                        <em style="color: #a0a0a0;">"Los shocks del pasado afectan el presente"</em></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #00DBDE;">📈 Aplicaciones Típicas</h4>
                <ul style="color: #e0e0e0; font-size: 0.9rem; line-height: 1.8;">
                    <li>🏭 Pronóstico de producción</li>
                    <li>📊 Predicción de ventas</li>
                    <li>🌡️ Modelado climático</li>
                    <li>📈 Análisis económico</li>
                    <li>🏥 Demanda de servicios de salud</li>
                    <li>⚡ Consumo energético</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Ecuación ARIMA con LaTeX
        st.latex(r"""
        Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t
        """)
        st.markdown("""
        <div style="text-align: center; color: #e0e0e0; margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
            <p style="margin: 0; font-size: 0.95rem;">
                Donde: <span style="color: #00DBDE;">φ = coeficientes AR</span>, 
                <span style="color: #FC00FF;">θ = coeficientes MA</span>, 
                <span style="color: #4ECDC4;">ε = errores</span>, 
                <span style="color: #FFD93D;">c = constante</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pestañas para cada parámetro
        tab1, tab2, tab3 = st.tabs(["📊 Parámetro p (AR)", "🔄 Parámetro d (I)", "📈 Parámetro q (MA)"])
        
        with tab1:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #00DBDE;">Parámetro p - Componente Autoregresivo (AR)</h4>
                <p style="color: #e0e0e0; line-height: 1.8;">
                    Representa el número de observaciones pasadas utilizadas para predecir el valor actual.
                </p>
                <div class="formula-box" style="margin-top: 1rem;">
                    <p style="font-size: 1.2rem; color: #00DBDE;">Y<sub>t</sub> = c + φ₁Y<sub>t-1</sub> + φ₂Y<sub>t-2</sub> + ... + φ<sub>p</sub>Y<sub>t-p</sub> + ε<sub>t</sub></p>
                </div>
                <h5 style="color: #4ECDC4; margin-top: 1.5rem;">✅ ¿Cómo identificar p?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Usa el gráfico <strong style="color: #00DBDE;">PACF</strong> (Autocorrelación Parcial)</li>
                    <li>Si PACF "corta" después del lag k → <strong style="color: #FC00FF;">p = k</strong></li>
                    <li>Ejemplo: Si PACF es significativa en lags 1 y 2, luego cae → p = 2</li>
                </ul>
                <div class="warning-box" style="margin-top: 1rem;">
                    <strong style="color: #FF6B6B;">⚠️ Precaución:</strong> Valores altos de p pueden causar sobreajuste.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #4ECDC4;">Parámetro d - Componente Integrado (I)</h4>
                <p style="color: #e0e0e0; line-height: 1.8;">
                    Número de veces que se debe diferenciar la serie para hacerla estacionaria.
                </p>
                <div class="formula-box" style="margin-top: 1rem;">
                    <p style="font-size: 1.2rem; color: #4ECDC4;">Primera diferencia: ∇Y<sub>t</sub> = Y<sub>t</sub> - Y<sub>t-1</sub></p>
                </div>
                <h5 style="color: #00DBDE; margin-top: 1.5rem;">✅ ¿Cómo identificar d?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Realiza la <strong style="color: #4ECDC4;">Prueba ADF</strong> (Augmented Dickey-Fuller)</li>
                    <li>Si p-valor > 0.05 → Serie no estacionaria → <strong style="color: #FC00FF;">d ≥ 1</strong></li>
                    <li>Si p-valor ≤ 0.05 → Serie estacionaria → <strong style="color: #FC00FF;">d = 0</strong></li>
                </ul>
                <div class="success-box" style="margin-top: 1rem;">
                    <strong style="color: #4ECDC4;">💡 Tip:</strong> La mayoría de series económicas requieren d=1.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #FC00FF;">Parámetro q - Componente de Media Móvil (MA)</h4>
                <p style="color: #e0e0e0; line-height: 1.8;">
                    Representa el número de términos de error pasados utilizados en el modelo.
                </p>
                <div class="formula-box" style="margin-top: 1rem;">
                    <p style="font-size: 1.2rem; color: #FC00FF;">Y<sub>t</sub> = μ + ε<sub>t</sub> + θ₁ε<sub>t-1</sub> + θ₂ε<sub>t-2</sub> + ... + θ<sub>q</sub>ε<sub>t-q</sub></p>
                </div>
                <h5 style="color: #4ECDC4; margin-top: 1.5rem;">✅ ¿Cómo identificar q?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Usa el gráfico <strong style="color: #FC00FF;">ACF</strong> (Autocorrelación)</li>
                    <li>Si ACF "corta" después del lag k → <strong style="color: #00DBDE;">q = k</strong></li>
                    <li>Ejemplo: Si ACF es significativa en lag 1, luego cae → q = 1</li>
                </ul>
                <div class="info-card" style="margin-top: 1rem;">
                    <strong style="color: #00DBDE;">📊 Interpretación:</strong> Los modelos MA son útiles para series con "shocks" temporales que se desvanecen.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Estacionalidad (SARIMA)
        st.markdown("""
        <div class="info-card" style="margin-top: 2rem;">
            <h4 style="color: #00DBDE;">🔄 SARIMA - ARIMA Estacional</h4>
            <p style="color: #e0e0e0; line-height: 1.8;">
                Cuando la serie tiene patrones que se repiten cada <strong style="color: #4ECDC4;">s</strong> periodos 
                (ej: cada 12 meses, cada 4 trimestres), usamos <strong style="color: #FC00FF;">SARIMA(p,d,q)(P,D,Q)s</strong>:
            </p>
            <div class="formula-box" style="margin-top: 1rem;">
                <p style="font-size: 1.1rem; color: #FC00FF;">
                    SARIMA(1,1,1)(1,1,1)<sub>12</sub> → ARIMA(1,1,1) + Componente estacional con periodo 12
                </p>
            </div>
            <ul style="color: #e0e0e0; line-height: 1.8; margin-top: 1rem;">
                <li><strong style="color: #00DBDE;">P, D, Q:</strong> Componentes estacionales AR, I, MA</li>
                <li><strong style="color: #4ECDC4;">s:</strong> Periodo estacional (12 para mensual, 4 para trimestral)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== SECCIÓN 2: ANÁLISIS EXPLORATORIO ====================
    elif opcion == "🔍 Análisis Exploratorio con Datasets Educativos":
        st.markdown('<h2 class="section-header">🔍 Análisis Exploratorio con Datasets Educativos</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-box">
            <h4 style="color: #4ECDC4;">📚 Datasets Educativos Disponibles</h4>
            <p style="color: #e0e0e0; line-height: 1.8;">
                En lugar de datos financieros, utilizaremos <strong style="color: #00DBDE;">datasets clásicos</strong> 
                de series temporales que muestran patrones educativos claros:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Selector de dataset
        dataset_name = st.selectbox(
            "Selecciona un dataset educativo:",
            list(DATASETS.keys()),
            help="Cada dataset muestra diferentes patrones de series temporales"
        )
        
        dataset_info = DATASETS[dataset_name]
        
        # Mostrar información del dataset
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color: #00DBDE;">📊 {dataset_name}</h4>
            <p style="color: #e0e0e0; line-height: 1.8;">{dataset_info['desc']}</p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                <div><strong style="color: #4ECDC4;">📅 Frecuencia:</strong> {dataset_info['freq']}</div>
                <div><strong style="color: #4ECDC4;">📆 Periodo:</strong> {dataset_info['period']}</div>
                <div><strong style="color: #4ECDC4;">🎯 Ideal para:</strong> {dataset_info['ideal_for']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cargar datos
        if st.button("📥 Cargar y Analizar Dataset", type="primary"):
            with st.spinner("🔄 Cargando datos..."):
                data = dataset_info['func']()
                
                if data is not None and len(data) > 50:
                    # Gráfico principal
                    st.markdown('<h3 class="subsection-header">📈 Visualización Completa</h3>', unsafe_allow_html=True)
                    
                    # Guía de interpretación
                    st.markdown("""
                    <div class="info-card">
                        <h5 style="color: #00DBDE;">📖 ¿Cómo interpretar estos gráficos?</h5>
                        <ul style="color: #e0e0e0; line-height: 1.8;">
                            <li><strong style="color: #4ECDC4;">Gráfico 1 (Serie Temporal):</strong> Muestra la evolución de los datos en el tiempo. 
                                Busca <span style="color: #00DBDE;">tendencia</span> (dirección general), <span style="color: #FC00FF;">estacionalidad</span> (patrones que se repiten) 
                                y <span style="color: #FF6B6B;">ciclos</span> (fluctuaciones irregulares).</li>
                            <li><strong style="color: #4ECDC4;">Gráfico 2 (Distribución):</strong> Histograma que muestra cómo se distribuyen los valores. 
                                Una forma de <span style="color: #00DBDE;">campana</span> indica normalidad. Valores extremos indican <span style="color: #FF6B6B;">outliers</span>.</li>
                            <li><strong style="color: #4ECDC4;">Gráfico 3 (Tendencia):</strong> Línea punteada muestra la media móvil (suavizado). 
                                Si sigue la tendencia general, confirma la <span style="color: #00DBDE;">dirección</span> de la serie.</li>
                            <li><strong style="color: #4ECDC4;">Gráfico 4 (Estacionalidad):</strong> Barras muestran el promedio por mes. 
                                Patrones regulares indican <span style="color: #FC00FF;">estacionalidad fuerte</span> (ej: ventas altas en diciembre).</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig1 = plot_time_series_advanced(data, dataset_name, dataset_info)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Métricas básicas
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Observaciones", f"{len(data):,}")
                    col2.metric("Media", f"{data.mean():.2f}")
                    col3.metric("Desviación", f"{data.std():.2f}")
                    col4.metric("Rango", f"{data.max() - data.min():.2f}")
                    
                    # Prueba de estacionariedad
                    st.markdown('<h3 class="subsection-header">📊 Prueba de Estacionariedad (ADF)</h3>', unsafe_allow_html=True)
                    adf_result, is_stationary = test_stationarity(data)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: {'#4ECDC4' if is_stationary else '#FF6B6B'}">
                                {adf_result['p-value']:.4f}
                            </div>
                            <div class="metric-label">p-valor ADF</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: {'#4ECDC4' if is_stationary else '#FF6B6B'}">
                                {adf_result['ADF Statistic']:.4f}
                            </div>
                            <div class="metric-label">ADF Statistic</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: {'#4ECDC4' if is_stationary else '#FF6B6B'}">
                                {adf_result['Critical Values']['5%']:.4f}
                            </div>
                            <div class="metric-label">Valor Crítico (5%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if is_stationary:
                        st.success("✅ La serie es **estacionaria** (p-valor ≤ 0.05). Podemos proceder con ARMA o ARIMA con d=0.")
                    else:
                        st.warning("❌ La serie **no es estacionaria** (p-valor > 0.05). Necesitamos diferenciar (d ≥ 1).")
                    
                    # ACF y PACF
                    st.markdown('<h3 class="subsection-header">📊 Gráficos ACF y PACF</h3>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="info-card">
                        <p style="color: #e0e0e0; line-height: 1.8;">
                            <strong style="color: #00DBDE;">ACF (Autocorrelación):</strong> Ayuda a identificar el parámetro <strong style="color: #FC00FF;">q (MA)</strong>. 
                            Busca dónde "corta" abruptamente.<br>
                            <strong style="color: #4ECDC4;">PACF (Autocorrelación Parcial):</strong> Ayuda a identificar el parámetro <strong style="color: #00DBDE;">p (AR)</strong>. 
                            Busca dónde "corta" abruptamente.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    fig2 = plot_acf_pacf_advanced(data, lags=min(40, len(data)//2))
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Diferenciación si es necesario
                    if not is_stationary:
                        st.markdown('<h3 class="subsection-header">🔄 Primera Diferencia</h3>', unsafe_allow_html=True)
                        diff_data = data.diff().dropna()
                        adf_diff, is_stationary_diff = test_stationarity(diff_data)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_diff = go.Figure()
                            fig_diff.add_trace(go.Scatter(
                                x=diff_data.index, y=diff_data.values,
                                mode='lines',
                                line=dict(color='#00DBDE', width=2),
                                name='Primera Diferencia'
                            ))
                            fig_diff.add_hline(y=0, line_color="#888888", line_dash="dash")
                            fig_diff.update_layout(
                                title="Primera Diferencia de la Serie",
                                template='plotly_dark',
                                height=300,
                                showlegend=False
                            )
                            st.plotly_chart(fig_diff, use_container_width=True)
                        
                        with col2:
                            if is_stationary_diff:
                                st.success("✅ Después de la primera diferencia, la serie es **estacionaria**. Usaremos d=1.")
                            else:
                                st.warning("❌ Aún no es estacionaria. Podríamos necesitar d=2.")
    
    # ==================== SECCIÓN 3: 5 EJEMPLOS ARIMA ====================
    elif opcion == "🎯 5 Ejemplos ARIMA Paso a Paso":
        st.markdown('<h2 class="section-header">🎯 5 Ejemplos ARIMA Paso a Paso</h2>', unsafe_allow_html=True)
        
        # Cargar dataset de ejemplo (AirPassengers por defecto)
        @st.cache_data
        def load_sample_data():
            return load_airpassengers()
        
        if 'sample_data' not in st.session_state:
            st.session_state.sample_data = load_sample_data()
        
        sample_data = st.session_state.sample_data
        
        # Dividir train/test
        train_size = int(len(sample_data) * 0.8)
        train, test = sample_data[:train_size], sample_data[train_size:]
        
        st.markdown(f"""
        <div class="tech-box">
            <h4 style="color: #4ECDC4;">📊 Dataset: AirPassengers</h4>
            <p style="color: #e0e0e0; line-height: 1.8;">
                Pasajeros aéreos mensuales (1949-1960). Mostraremos 5 configuraciones ARIMA diferentes 
                para que compares cómo cada una captura los patrones de la serie.
            </p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                <div><strong style="color: #00DBDE;">📈 Entrenamiento:</strong> {len(train)} observaciones</div>
                <div><strong style="color: #00DBDE;">🎯 Prueba:</strong> {len(test)} observaciones</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 5 ejemplos ARIMA
        examples = [
            {"p": 1, "d": 1, "q": 1, "nombre": "ARIMA(1,1,1) - Modelo Simple", "desc": "Configuración básica que captura tendencia y algo de autocorrelación."},
            {"p": 2, "d": 1, "q": 1, "nombre": "ARIMA(2,1,1) - Más AR", "desc": "Agrega un término AR adicional para mejor captura de inercia."},
            {"p": 1, "d": 1, "q": 2, "nombre": "ARIMA(1,1,2) - Más MA", "desc": "Agrega un término MA adicional para mejor manejo de shocks."},
            {"p": 2, "d": 1, "q": 2, "nombre": "ARIMA(2,1,2) - Balanceado", "desc": "Combinación equilibrada de componentes AR y MA."},
            {"p": 3, "d": 1, "q": 3, "nombre": "ARIMA(3,1,3) - Complejo", "desc": "Modelo más complejo que puede capturar patrones adicionales (cuidado con sobreajuste)."}
        ]
        
        for i, example in enumerate(examples, 1):
            with st.expander(f"📌 Ejemplo {i}: {example['nombre']}", expanded=(i==1)):
                st.markdown(f"""
                <div class="info-card">
                    <p style="color: #e0e0e0; line-height: 1.8;">{example['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                order = (example['p'], example['d'], example['q'])
                try:
                    model_fit = fit_arima_model(train, order)
                    if model_fit:
                        evaluation = evaluate_model(model_fit, test)
                        if evaluation:
                            # Métricas
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("RMSE", f"{evaluation['RMSE']:.2f}")
                            col2.metric("MAPE", f"{evaluation['MAPE']:.1f}%")
                            col3.metric("AIC", f"{model_fit.aic:.0f}")
                            col4.metric("BIC", f"{model_fit.bic:.0f}")
                            
                            # Gráfico de pronóstico
                            fig_forecast = plot_forecast_advanced(
                                train, test, evaluation['forecast'], model_fit, "AirPassengers", order
                            )
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            # Análisis de residuos
                            st.markdown('<h4 style="color: #00DBDE;">📊 Análisis de Residuos</h4>', unsafe_allow_html=True)
                            residuals = model_fit.resid
                            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1a2e')
                            
                            # Histograma
                            ax1.hist(residuals, bins=30, density=True, alpha=0.7, color='#00DBDE', edgecolor='#4ECDC4')
                            ax1.axvline(x=0, color='#FC00FF', linestyle='--', alpha=0.8, linewidth=2)
                            ax1.set_title('Distribución de Residuos', color='white', fontsize=14, fontweight='bold')
                            ax1.set_facecolor('#1a1a2e')
                            ax1.tick_params(colors='white')
                            
                            # Q-Q Plot
                            stats.probplot(residuals, dist="norm", plot=ax2)
                            ax2.set_title('Q-Q Plot (Normalidad)', color='white', fontsize=14, fontweight='bold')
                            ax2.tick_params(colors='white')
                            for line in ax2.lines:
                                line.set_color('#4ECDC4')
                            for scatter in ax2.collections:
                                scatter.set_color('#00DBDE')
                            
                            plt.tight_layout()
                            st.pyplot(fig2)
                            
                            # Interpretación
                            st.markdown("""
                            <div class="success-box">
                                <h5 style="color: #4ECDC4;">💡 Interpretación</h5>
                                <ul style="color: #e0e0e0; line-height: 1.8;">
                                    <li><strong style="color: #00DBDE;">RMSE:</strong> Error cuadrático medio. Cuanto menor, mejor.</li>
                                    <li><strong style="color: #4ECDC4;">MAPE:</strong> Error porcentual medio. < 10% es excelente, < 20% es bueno.</li>
                                    <li><strong style="color: #FC00FF;">AIC/BIC:</strong> Criterios de información. Menor es mejor (penaliza complejidad).</li>
                                    <li><strong style="color: #FF6B6B;">Residuos:</strong> Deben parecerse a ruido blanco (media 0, varianza constante, normales).</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error al ajustar el modelo: {e}")
    
    # ==================== SECCIÓN 4: COMPARATIVA DE MODELOS ====================
    elif opcion == "📊 Comparativa de Modelos":
        st.markdown('<h2 class="section-header">📊 Comparativa de Modelos</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-box">
            <h4 style="color: #4ECDC4;">🔬 Experimento: Variar p, d, q</h4>
            <p style="color: #e0e0e0; line-height: 1.8;">
                Selecciona diferentes valores para los parámetros ARIMA y compara el rendimiento 
                de cada configuración en el dataset AirPassengers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cargar datos
        @st.cache_data
        def load_comparison_data():
            return load_airpassengers()
        
        if 'comparison_data' not in st.session_state:
            st.session_state.comparison_data = load_comparison_data()
        
        data = st.session_state.comparison_data
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]
        
        # Selectores de parámetros
        col1, col2, col3 = st.columns(3)
        with col1:
            p_values = st.multiselect("p (AR)", [0, 1, 2, 3, 4], default=[1, 2])
        with col2:
            d_values = st.multiselect("d (I)", [0, 1, 2], default=[1])
        with col3:
            q_values = st.multiselect("q (MA)", [0, 1, 2, 3, 4], default=[1, 2])
        
        if st.button("🔄 Ejecutar Comparativa", type="primary"):
            with st.spinner("⏳ Ejecutando múltiples modelos..."):
                results = []
                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            try:
                                order = (p, d, q)
                                model_fit = fit_arima_model(train, order)
                                if model_fit:
                                    evaluation = evaluate_model(model_fit, test)
                                    if evaluation:
                                        results.append({
                                            'Modelo': f'ARIMA({p},{d},{q})',
                                            'RMSE': evaluation['RMSE'],
                                            'MAE': evaluation['MAE'],
                                            'MAPE': evaluation['MAPE'],
                                            'AIC': model_fit.aic,
                                            'BIC': model_fit.bic
                                        })
                            except:
                                continue
                
                if results:
                    results_df = pd.DataFrame(results).sort_values('RMSE')
                    
                    # Tabla de resultados
                    st.markdown('<h3 class="subsection-header">📋 Resultados Ordenados por RMSE</h3>', unsafe_allow_html=True)
                    display_df = results_df.copy()
                    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f'{x:.2f}')
                    display_df['MAE'] = display_df['MAE'].apply(lambda x: f'{x:.2f}')
                    display_df['MAPE'] = display_df['MAPE'].apply(lambda x: f'{x:.1f}%')
                    display_df['AIC'] = display_df['AIC'].apply(lambda x: f'{x:.0f}')
                    display_df['BIC'] = display_df['BIC'].apply(lambda x: f'{x:.0f}')
                    
                    st.dataframe(
                        display_df.style.set_properties(**{
                            'background-color': 'rgba(255,255,255,0.1)',
                            'color': '#e0e0e0',
                            'border-color': 'rgba(0,219,222,0.3)'
                        }),
                        use_container_width=True
                    )
                    
                    # Gráficos comparativos
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('RMSE (Error Cuadrático)', 'MAPE (%)', 'AIC', 'MAE'),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1
                    )
                    
                    colors_list = ['#00DBDE', '#4ECDC4', '#FC00FF', '#FF6B6B']
                    metrics = [('RMSE', 1, 1), ('MAPE', 1, 2), ('AIC', 2, 1), ('MAE', 2, 2)]
                    
                    for idx, (metric, row, col) in enumerate(metrics):
                        fig.add_trace(
                            go.Bar(
                                x=results_df['Modelo'],
                                y=results_df[metric],
                                marker_color=colors_list[idx % len(colors_list)],
                                text=results_df[metric].apply(lambda x: f'{x:.2f}'),
                                textposition='auto'
                            ),
                            row=row, col=col
                        )
                    
                    fig.update_layout(
                        height=600,
                        template='plotly_dark',
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0', size=11)
                    )
                    fig.update_xaxes(tickangle=45, gridcolor='rgba(255,255,255,0.1)')
                    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mejor modelo
                    best = results_df.iloc[0]
                    st.markdown(f"""
                    <div class="success-box">
                        <h5 style="color: #4ECDC4;">💡 Mejor Modelo</h5>
                        <p style="color: #e0e0e0; font-size: 1.1rem;">
                            El modelo con mejor rendimiento es <strong style="color: #00DBDE;">{best['Modelo']}</strong> 
                            con RMSE = {best['RMSE']:.2f} y MAPE = {best['MAPE']:.1f}%.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No se pudieron ajustar modelos con los parámetros seleccionados.")
    
    # ==================== SECCIÓN 5: LABORATORIO EXPERIMENTAL ====================
    elif opcion == "🧪 Laboratorio Experimental":
        st.markdown('<h2 class="section-header">🧪 Laboratorio Experimental</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-box">
            <h4 style="color: #4ECDC4;">🔬 Experimenta con Tus Propios Parámetros</h4>
            <p style="color: #e0e0e0; line-height: 1.8;">
                Selecciona un dataset, ajusta los parámetros ARIMA y observa los resultados en tiempo real.
                ¡Ideal para entender cómo cada parámetro afecta el pronóstico!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Selector de dataset
        lab_dataset = st.selectbox(
            "Selecciona un dataset:",
            list(DATASETS.keys()),
            key="lab_dataset"
        )
        
        # Sliders para parámetros
        col1, col2, col3 = st.columns(3)
        with col1:
            lab_p = st.slider("p (AR) - Términos autoregresivos", 0, 5, 1)
        with col2:
            lab_d = st.slider("d (I) - Órdenes de diferenciación", 0, 3, 1)
        with col3:
            lab_q = st.slider("q (MA) - Términos de media móvil", 0, 5, 1)
        
        # Porcentaje de entrenamiento
        train_pct = st.slider("Porcentaje para entrenamiento", 50, 95, 80, 5)
        
        if st.button("🚀 Ejecutar ARIMA", type="primary"):
            with st.spinner(f"⏳ Ajustando ARIMA({lab_p},{lab_d},{lab_q})..."):
                # Cargar datos
                data = DATASETS[lab_dataset]['func']()
                
                if data is not None and len(data) > 100:
                    train_size = int(len(data) * train_pct / 100)
                    train, test = data[:train_size], data[train_size:]
                    
                    try:
                        order = (lab_p, lab_d, lab_q)
                        model_fit = fit_arima_model(train, order)
                        
                        if model_fit:
                            evaluation = evaluate_model(model_fit, test)
                            
                            if evaluation:
                                # Métricas
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("RMSE", f"{evaluation['RMSE']:.2f}")
                                col2.metric("MAE", f"{evaluation['MAE']:.2f}")
                                col3.metric("MAPE", f"{evaluation['MAPE']:.2f}%")
                                col4.metric("AIC", f"{model_fit.aic:.2f}")
                                
                                # Gráfico de pronóstico
                                fig = plot_forecast_advanced(
                                    train, test, evaluation['forecast'], model_fit, lab_dataset, order
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Descargar pronóstico
                                forecast_df = pd.DataFrame({
                                    'Fecha': pd.date_range(
                                        start=train.index[-1] + pd.Timedelta(days=1),
                                        periods=len(test),
                                        freq='MS'
                                    ),
                                    'Pronóstico': evaluation['forecast']
                                })
                                
                                st.download_button(
                                    label="📥 Descargar Pronóstico (CSV)",
                                    data=forecast_df.to_csv(index=False),
                                    file_name=f"pronostico_{lab_dataset.replace(' ', '_')}_arima{order}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("❌ No se pudo evaluar el modelo.")
                        else:
                            st.error("❌ No se pudo ajustar el modelo ARIMA. Intenta con otros parámetros.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.error("❌ Datos insuficientes. Selecciona otro dataset.")
    
    # ==================== SECCIÓN 6: AR vs MA vs ARMA vs ARIMA vs SARIMA ====================
    elif opcion == "🔄 AR vs MA vs ARMA vs ARIMA vs SARIMA":
        st.markdown('<h2 class="section-header">🔄 Comparativa: AR vs MA vs ARMA vs ARIMA vs SARIMA</h2>', unsafe_allow_html=True)
        
        # Tabla comparativa
        comparison_data = {
            'Modelo': ['AR(p)', 'MA(q)', 'ARMA(p,q)', 'ARIMA(p,d,q)', 'SARIMA(p,d,q)(P,D,Q)s'],
            'Nombre Completo': ['Autoregresivo', 'Media Móvil', 'AR + MA', 'ARIMA', 'ARIMA Estacional'],
            '¿Requiere Estacionariedad?': ['✅ Sí', '✅ Sí', '✅ Sí', '❌ No (diferencia)', '❌ No (diferencia)'],
            'Uso Principal': ['Inercia/Persistencia', 'Shocks temporales', 'Serie estacionaria compleja', 'Serie con tendencia', 'Serie con estacionalidad'],
            'Ejemplo': ['Temperatura diaria', 'Ventas post-desastre', 'Rendimientos financieros', 'PIB trimestral', 'Ventas navideñas']
        }
        
        st.dataframe(
            pd.DataFrame(comparison_data).style.set_properties(**{
                'background-color': 'rgba(255,255,255,0.1)',
                'color': '#e0e0e0',
                'border-color': 'rgba(0,219,222,0.3)'
            }),
            use_container_width=True,
            height=300
        )
        
        # Pestañas para cada modelo
        tab_ar, tab_ma, tab_arma, tab_arima, tab_sarima = st.tabs(["AR", "MA", "ARMA", "ARIMA", "SARIMA"])
        
        with tab_ar:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #00DBDE;">AR(p) - Modelo Autoregresivo</h4>
                <div class="formula-box">
                    <p style="font-size: 1.3rem; color: #00DBDE;">Y<sub>t</sub> = c + φ₁Y<sub>t-1</sub> + ... + φ<sub>p</sub>Y<sub>t-p</sub> + ε<sub>t</sub></p>
                </div>
                <h5 style="color: #4ECDC4; margin-top: 1.5rem;">✅ ¿Cuándo usar AR?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Cuando hay <strong style="color: #00DBDE;">inercia o persistencia</strong> en la serie</li>
                    <li>Cuando el <strong style="color: #4ECDC4;">PACF corta abruptamente</strong> después de p lags</li>
                    <li>Ejemplo: Temperatura, donde hoy depende fuertemente de ayer</li>
                </ul>
                <div class="warning-box" style="margin-top: 1rem;">
                    <strong style="color: #FF6B6B;">⚠️ Limitación:</strong> Solo funciona con series estacionarias.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab_ma:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #FC00FF;">MA(q) - Modelo de Media Móvil</h4>
                <div class="formula-box">
                    <p style="font-size: 1.3rem; color: #FC00FF;">Y<sub>t</sub> = μ + ε<sub>t</sub> + θ₁ε<sub>t-1</sub> + ... + θ<sub>q</sub>ε<sub>t-q</sub></p>
                </div>
                <h5 style="color: #4ECDC4; margin-top: 1.5rem;">✅ ¿Cuándo usar MA?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Cuando hay <strong style="color: #FC00FF;">shocks temporales</strong> que se desvanecen</li>
                    <li>Cuando el <strong style="color: #4ECDC4;">ACF corta abruptamente</strong> después de q lags</li>
                    <li>Ejemplo: Ventas después de un evento único (desastre natural)</li>
                </ul>
                <div class="info-card" style="margin-top: 1rem;">
                    <strong style="color: #00DBDE;">💡 Tip:</strong> Los modelos MA son menos intuitivos pero útiles para series con "memoria corta".
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab_arma:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #4ECDC4;">ARMA(p,q) - Combinación AR + MA</h4>
                <div class="formula-box">
                    <p style="font-size: 1.3rem; color: #4ECDC4;">Y<sub>t</sub> = c + Σφ<sub>i</sub>Y<sub>t-i</sub> + Σθ<sub>j</sub>ε<sub>t-j</sub> + ε<sub>t</sub></p>
                </div>
                <h5 style="color: #00DBDE; margin-top: 1.5rem;">✅ ¿Cuándo usar ARMA?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Cuando la serie es <strong style="color: #4ECDC4;">estacionaria</strong> pero compleja</li>
                    <li>Cuando <strong style="color: #00DBDE;">tanto ACF como PACF decaen gradualmente</strong></li>
                    <li>Ejemplo: Rendimientos de acciones (sin tendencia)</li>
                </ul>
                <div class="warning-box" style="margin-top: 1rem;">
                    <strong style="color: #FF6B6B;">⚠️ Precaución:</strong> No usar con series que tienen tendencia.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab_arima:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #00DBDE;">ARIMA(p,d,q) - ARIMA con Diferenciación</h4>
                <div class="formula-box">
                    <p style="font-size: 1.3rem; color: #00DBDE;">ARMA aplicado a datos diferenciados d veces</p>
                </div>
                <h5 style="color: #4ECDC4; margin-top: 1.5rem;">✅ ¿Cuándo usar ARIMA?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Cuando la serie tiene <strong style="color: #00DBDE;">tendencia</strong> (no estacionaria)</li>
                    <li>La <strong style="color: #4ECDC4;">mayoría de series económicas y financieras</strong></li>
                    <li>Ejemplo: PIB, ventas, precios de acciones</li>
                </ul>
                <div class="success-box" style="margin-top: 1rem;">
                    <strong style="color: #4ECDC4;">💡 Ventaja:</strong> Maneja tendencias sin necesidad de modelarlas explícitamente.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab_sarima:
            st.markdown("""
            <div class="tech-box">
                <h4 style="color: #4ECDC4;">SARIMA(p,d,q)(P,D,Q)s - ARIMA Estacional</h4>
                <div class="formula-box">
                    <p style="font-size: 1.3rem; color: #4ECDC4;">ARIMA + Componentes estacionales (P,D,Q) con periodo s</p>
                </div>
                <h5 style="color: #00DBDE; margin-top: 1.5rem;">✅ ¿Cuándo usar SARIMA?</h5>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Cuando hay <strong style="color: #4ECDC4;">patrones estacionales claros</strong></li>
                    <li><strong style="color: #00DBDE;">Ventas minoristas</strong> (picos navideños)</li>
                    <li><strong style="color: #00DBDE;">Turismo</strong> (temporadas altas/bajas)</li>
                    <li><strong style="color: #00DBDE;">Consumo energético</strong> (verano/invierno)</li>
                </ul>
                <div class="info-card" style="margin-top: 1rem;">
                    <strong style="color: #00DBDE;">📊 Ejemplo:</strong> SARIMA(1,1,1)(1,1,1)<sub>12</sub> para datos mensuales con estacionalidad anual.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Ejemplo práctico comparativo
        st.markdown('<h3 class="subsection-header">🧪 Ejemplo Práctico Comparativo</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            comp_dataset = st.selectbox(
                "Dataset para comparar:",
                list(DATASETS.keys()),
                key="comp_dataset"
            )
        
        if st.button("🔄 Comparar Modelos", type="primary", key="comp_button"):
            with st.spinner("⏳ Ajustando múltiples modelos..."):
                data = DATASETS[comp_dataset]['func']()
                
                if data is not None and len(data) > 200:
                    train_size = int(len(data) * 0.8)
                    train, test = data[:train_size], data[train_size:]
                    
                    models_results = []
                    
                    # AR(2)
                    try:
                        m = ARIMA(train, order=(2, 0, 0)).fit()
                        e = evaluate_model(m, test)
                        if e:
                            models_results.append({'Modelo': 'AR(2)', 'Tipo': 'AR', 'RMSE': e['RMSE'], 'MAPE': e['MAPE'], 'AIC': m.aic, 'BIC': m.bic})
                    except:
                        pass
                    
                    # MA(2)
                    try:
                        m = ARIMA(train, order=(0, 0, 2)).fit()
                        e = evaluate_model(m, test)
                        if e:
                            models_results.append({'Modelo': 'MA(2)', 'Tipo': 'MA', 'RMSE': e['RMSE'], 'MAPE': e['MAPE'], 'AIC': m.aic, 'BIC': m.bic})
                    except:
                        pass
                    
                    # ARMA(2,2)
                    try:
                        m = ARIMA(train, order=(2, 0, 2)).fit()
                        e = evaluate_model(m, test)
                        if e:
                            models_results.append({'Modelo': 'ARMA(2,2)', 'Tipo': 'ARMA', 'RMSE': e['RMSE'], 'MAPE': e['MAPE'], 'AIC': m.aic, 'BIC': m.bic})
                    except:
                        pass
                    
                    # ARIMA(2,1,2)
                    try:
                        m = ARIMA(train, order=(2, 1, 2)).fit()
                        e = evaluate_model(m, test)
                        if e:
                            models_results.append({'Modelo': 'ARIMA(2,1,2)', 'Tipo': 'ARIMA', 'RMSE': e['RMSE'], 'MAPE': e['MAPE'], 'AIC': m.aic, 'BIC': m.bic})
                    except:
                        pass
                    
                    # SARIMA
                    try:
                        m = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        fc = m.get_forecast(steps=len(test)).predicted_mean
                        rmse = np.sqrt(mean_squared_error(test, fc))
                        mape = np.mean(np.abs((test - fc) / test)) * 100
                        models_results.append({'Modelo': 'SARIMA(1,1,1)(1,1,1,12)', 'Tipo': 'SARIMA', 'RMSE': rmse, 'MAPE': mape, 'AIC': m.aic, 'BIC': m.bic})
                    except:
                        pass
                    
                    if models_results:
                        results_df = pd.DataFrame(models_results).sort_values('RMSE')
                        
                        # Mostrar resultados
                        display = results_df.copy()
                        display['RMSE'] = display['RMSE'].apply(lambda x: f'{x:.2f}')
                        display['MAPE'] = display['MAPE'].apply(lambda x: f'{x:.1f}%')
                        display['AIC'] = display['AIC'].apply(lambda x: f'{x:.0f}')
                        display['BIC'] = display['BIC'].apply(lambda x: f'{x:.0f}')
                        
                        st.dataframe(
                            display.style.set_properties(**{
                                'background-color': 'rgba(255,255,255,0.1)',
                                'color': '#e0e0e0',
                                'border-color': 'rgba(0,219,222,0.3)'
                            }),
                            use_container_width=True
                        )
                        
                        # Gráficos comparativos
                        colors_map = {
                            'AR(2)': '#00DBDE',
                            'MA(2)': '#4ECDC4',
                            'ARMA(2,2)': '#FC00FF',
                            'ARIMA(2,1,2)': '#FF6B6B',
                            'SARIMA(1,1,1)(1,1,1,12)': '#FFD93D'
                        }
                        
                        fig = make_subplots(rows=2, cols=2, subplot_titles=('RMSE', 'MAPE', 'AIC', 'BIC'))
                        
                        for metric, row, col in [('RMSE', 1, 1), ('MAPE', 1, 2), ('AIC', 2, 1), ('BIC', 2, 2)]:
                            for _, r in results_df.iterrows():
                                fig.add_trace(
                                    go.Bar(
                                        x=[r['Modelo']],
                                        y=[r[metric]],
                                        marker_color=colors_map.get(r['Modelo'], '#888888'),
                                        showlegend=False,
                                        text=[f'{r[metric]:.2f}'],
                                        textposition='auto'
                                    ),
                                    row=row, col=col
                                )
                        
                        fig.update_layout(
                            height=600,
                            template='plotly_dark',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e0e0e0', size=11)
                        )
                        fig.update_xaxes(tickangle=45, gridcolor='rgba(255,255,255,0.1)')
                        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mejor modelo
                        best = results_df.iloc[0]
                        st.markdown(f"""
                        <div class="success-box">
                            <h5 style="color: #4ECDC4;">💡 Conclusión</h5>
                            <p style="color: #e0e0e0; font-size: 1.1rem;">
                                El mejor modelo para {comp_dataset} es <strong style="color: #00DBDE;">{best['Modelo']}</strong> 
                                con RMSE = {best['RMSE']:.2f} y MAPE = {best['MAPE']:.1f}%.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No se pudieron ajustar modelos.")
                else:
                    st.error("❌ Datos insuficientes.")
        
        # Guía rápida
        st.markdown('<h3 class="subsection-header">🎯 Guía Rápida de Selección</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem;">
            <div class="info-card">
                <h5 style="color: #00DBDE;">📊 Si la serie es ESTACIONARIA:</h5>
                <ul style="color: #e0e0e0; font-size: 0.95rem; line-height: 1.8;">
                    <li>ACF decae, PACF corta → <span style="color: #00DBDE; font-weight: bold;">AR(p)</span></li>
                    <li>PACF decae, ACF corta → <span style="color: #4ECDC4; font-weight: bold;">MA(q)</span></li>
                    <li>Ambos decaen → <span style="color: #FC00FF; font-weight: bold;">ARMA(p,q)</span></li>
                </ul>
            </div>
            <div class="info-card">
                <h5 style="color: #4ECDC4;">📈 Si NO es estacionaria:</h5>
                <ul style="color: #e0e0e0; font-size: 0.95rem; line-height: 1.8;">
                    <li>Solo tendencia → <span style="color: #FF6B6B; font-weight: bold;">ARIMA(p,d,q)</span></li>
                    <li>Tendencia + estacionalidad → <span style="color: #FFD93D; font-weight: bold;">SARIMA(p,d,q)(P,D,Q)s</span></li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== SECCIÓN 7: LIMITACIONES Y CASOS DE USO ====================
    elif opcion == "⚠️ Limitaciones y Casos de Uso":
        st.markdown('<h2 class="section-header">⚠️ Limitaciones y Casos de Uso de ARIMA</h2>', unsafe_allow_html=True)
        
        # ¿Cuándo usar ARIMA?
        st.markdown("""
        <div class="success-box">
            <h4 style="color: #4ECDC4;">✅ ¿Cuándo USAR ARIMA?</h4>
            <ul style="color: #e0e0e0; line-height: 2; font-size: 1.05rem;">
                <li>📊 <strong style="color: #00DBDE;">Series univariadas</strong> con patrones claros (tendencia, estacionalidad)</li>
                <li>📈 <strong style="color: #00DBDE;">Pronósticos a corto plazo</strong> (pocos periodos adelante)</li>
                <li>📉 <strong style="color: #00DBDE;">Series con comportamiento estable</strong> en el tiempo</li>
                <li>🏭 <strong style="color: #00DBDE;">Producción, ventas, demanda</strong> con patrones históricos</li>
                <li>🌡️ <strong style="color: #00DBDE;">Datos climáticos</strong> con ciclos predecibles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # ¿Cuándo NO usar ARIMA?
        st.markdown("""
        <div class="warning-box">
            <h4 style="color: #FF6B6B;">❌ ¿Cuándo NO USAR ARIMA?</h4>
            <ul style="color: #e0e0e0; line-height: 2; font-size: 1.05rem;">
                <li>📊 <strong style="color: #FF6B6B;">Series multivariadas</strong> (múltiples variables relacionadas)</li>
                <li>📉 <strong style="color: #FF6B6B;">Pronósticos a largo plazo</strong> (más de 10-20 periodos)</li>
                <li>🔄 <strong style="color: #FF6B6B;">Series con cambios estructurales</strong> (quiebres, crisis)</li>
                <li>💥 <strong style="color: #FF6B6B;">Eventos extremos</strong> no presentes en el histórico</li>
                <li>📱 <strong style="color: #FF6B6B;">Datos de redes sociales</strong> (muy volátiles, sin patrones)</li>
                <li>🚀 <strong style="color: #FF6B6B;">Tecnologías emergentes</strong> (sin histórico suficiente)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Limitaciones importantes
        st.markdown("""
        <div class="tech-box">
            <h4 style="color: #00DBDE;">⚠️ Limitaciones Importantes de ARIMA</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin-top: 1rem;">
                <div class="info-card">
                    <h5 style="color: #FC00FF;">1. Linealidad</h5>
                    <p style="color: #e0e0e0; font-size: 0.95rem;">ARIMA asume relaciones lineales. No captura patrones no lineales complejos.</p>
                </div>
                <div class="info-card">
                    <h5 style="color: #FC00FF;">2. Estacionariedad</h5>
                    <p style="color: #e0e0e0; font-size: 0.95rem;">Requiere series estacionarias (o diferenciables). No maneja tendencias exponenciales bien.</p>
                </div>
                <div class="info-card">
                    <h5 style="color: #FC00FF;">3. Univariable</h5>
                    <p style="color: #e0e0e0; font-size: 0.95rem;">Solo usa el histórico de la serie. No incorpora variables externas (precio, marketing, etc.).</p>
                </div>
                <div class="info-card">
                    <h5 style="color: #FC00FF;">4. Corto Plazo</h5>
                    <p style="color: #e0e0e0; font-size: 0.95rem;">Pronósticos a largo plazo tienden a la media. Pierden precisión rápidamente.</p>
                </div>
                <div class="info-card">
                    <h5 style="color: #FC00FF;">5. Datos Históricos</h5>
                    <p style="color: #e0e0e0; font-size: 0.95rem;">Asume que el futuro será como el pasado. No anticipa cambios estructurales.</p>
                </div>
                <div class="info-card">
                    <h5 style="color: #FC00FF;">6. Outliers</h5>
                    <p style="color: #e0e0e0; font-size: 0.95rem;">Sensible a valores atípicos. Pueden distorsionar significativamente el modelo.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternativas a ARIMA
        st.markdown("""
        <div class="info-card" style="margin-top: 2rem;">
            <h4 style="color: #4ECDC4;">🔄 Alternativas a ARIMA</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                <div>
                    <h5 style="color: #00DBDE;">🧠 Redes Neuronales (LSTM)</h5>
                    <p style="color: #e0e0e0; font-size: 0.9rem;">Para patrones no lineales complejos y series multivariadas.</p>
                </div>
                <div>
                    <h5 style="color: #00DBDE;">🌲 Random Forest / XGBoost</h5>
                    <p style="color: #e0e0e0; font-size: 0.9rem;">Cuando hay múltiples variables explicativas.</p>
                </div>
                <div>
                    <h5 style="color: #00DBDE;">📊 Prophet (Facebook)</h5>
                    <p style="color: #e0e0e0; font-size: 0.9rem;">Para series con estacionalidad múltiple y holidays.</p>
                </div>
                <div>
                    <h5 style="color: #00DBDE;">📈 VAR / VECM</h5>
                    <p style="color: #e0e0e0; font-size: 0.9rem;">Para múltiples series temporales relacionadas.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Consejos prácticos
        st.markdown("""
        <div class="tech-box" style="margin-top: 2rem;">
            <h4 style="color: #4ECDC4;">💡 Consejos Prácticos</h4>
            <ul style="color: #e0e0e0; line-height: 2; font-size: 1.05rem;">
                <li>🔍 <strong style="color: #00DBDE;">Explora los datos</strong> antes de modelar (gráficos, estacionariedad, outliers)</li>
                <li>📊 <strong style="color: #00DBDE;">Divide en train/test</strong> para validar el modelo</li>
                <li>🎯 <strong style="color: #00DBDE;">Comienza simple</strong> (ARIMA(1,1,1)) y luego complejiza</li>
                <li>📉 <strong style="color: #00DBDE;">Revisa los residuos</strong> (deben ser ruido blanco)</li>
                <li>🔄 <strong style="color: #00DBDE;">Actualiza el modelo</strong> periódicamente con nuevos datos</li>
                <li>⚖️ <strong style="color: #00DBDE;">Considera el trade-off</strong> entre complejidad y precisión</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #a0a0a0; padding: 2rem; 
                background: linear-gradient(135deg, rgba(0,219,222,0.1) 0%, rgba(252,0,255,0.1) 100%); 
                border-radius: 15px; margin-top: 2rem;'>
        <h4 style="color: #00DBDE; font-size: 1.3rem;">📊 Dashboard Educativo de Modelos ARIMA</h4>
        <p style="color: #e0e0e0; margin: 0.5rem 0;">Desarrollado con Streamlit, Statsmodels y Plotly - Fines educativos</p>
        <p style="color: #a0a0a0; font-size: 0.9rem; margin: 0.5rem 0;">
            Datos: Datasets educativos clásicos (AirPassengers, CO2, etc.) | """ + datetime.now().strftime("%Y-%m-%d") + """
        </p>
        <p style="color: #FC00FF; font-size: 0.85rem; margin-top: 1rem;">
            ✨ Recuerda: ARIMA es una herramienta poderosa pero tiene limitaciones. 
            ¡Úsala sabiamente! ✨
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()