# 📈 Dashboard Interactivo de Modelos ARIMA

Un dashboard educativo desarrollado con **Streamlit** para entender y experimentar con modelos **ARIMA** (AutoRegressive Integrated Moving Average) utilizando datos reales del mercado financiero.

## 🎯 Objetivos

- **Educar** sobre los fundamentos de los modelos ARIMA
- **Demostrar** aplicaciones prácticas con datos reales de Yahoo Finance
- **Permitir** experimentación interactiva con diferentes parámetros
- **Visualizar** resultados de manera clara y comprensible

## 🚀 Características

### 5 Secciones Principales:

1. **📚 Introducción Teórica**
   - Explicación detallada de los componentes ARIMA
   - Fórmulas matemáticas y conceptos fundamentales
   - Guía para seleccionar parámetros (p, d, q)

2. **🔍 Análisis Exploratorio**
   - Carga de datos de cualquier activo de Yahoo Finance
   - Prueba de estacionariedad (Dickey-Fuller)
   - Gráficas ACF y PACF para identificar parámetros
   - Diferenciación de series temporales

3. **🎯 Ejemplos ARIMA**
   - 5 ejemplos con diferentes combinaciones de parámetros:
     - ARIMA(1,1,1) - Modelo Simple
     - ARIMA(2,1,1) - Más Autoregresivo
     - ARIMA(1,1,2) - Más Media Móvil
     - ARIMA(2,1,2) - Modelo Complejo
     - ARIMA(3,1,3) - Modelo Avanzado
   - Métricas de evaluación (RMSE, MAE, MAPE, AIC)
   - Diagnóstico de residuos

4. **📊 Comparativa de Modelos**
   - Comparación múltiple de modelos ARIMA
   - Selección de rangos de parámetros
   - Visualización de mejores modelos
   - Gráficas comparativas de pronósticos

5. **🧪 Laboratorio Interactivo**
   - Experimentación libre con parámetros
   - Selección de cualquier activo financiero
   - Ajuste de período de entrenamiento/prueba
   - Descarga de resultados en CSV

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (administrador de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   cd arima_dashboard
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**
   ```bash
   streamlit run app.py
   ```

5. **Acceder al dashboard**
   - La aplicación se abrirá automáticamente en tu navegador
   - URL local: `http://localhost:8501`

## 📊 Datos Utilizados

El dashboard utiliza datos históricos de **Yahoo Finance**, incluyendo:
- Precios de cierre ajustados
- Datos de acciones (AAPL, GOOGL, MSFT, TSLA, etc.)
- Índices bursátiles (^GSPC, ^DJI, ^IXIC)
- ETFs y otros instrumentos financieros

## 🎨 Tecnologías Utilizadas

- **Streamlit**: Framework para aplicaciones web de datos
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Cálculo numérico
- **Statsmodels**: Modelos estadísticos y series temporales
- **Plotly**: Gráficas interactivas
- **Matplotlib**: Gráficas estáticas
- **yfinance**: Descarga de datos financieros
- **Scikit-learn**: Métricas de evaluación

## 📖 Conceptos Clave ARIMA

### ¿Qué es ARIMA?

ARIMA (**A**uto**R**egressive **I**ntegrated **M**oving **A**verage) es un modelo estadístico para análisis y pronóstico de series temporales.

### Parámetros:

- **p (AR - Autoregresivo)**: Número de observaciones pasadas utilizadas como predictores
- **d (I - Integrado)**: Número de diferencias necesarias para hacer la serie estacionaria
- **q (MA - Media Móvil)**: Número de términos de error pasados utilizados

### Fórmula General:

```
Y_t = c + Σ(φ_i * Y_{t-i}) + Σ(θ_j * ε_{t-j}) + ε_t
```

Donde:
- `Y_t` es el valor en el tiempo t
- `c` es una constante
- `φ_i` son los coeficientes autoregresivos
- `θ_j` son los coeficientes de media móvil
- `ε_t` es el error en el tiempo t

## 🔍 Métricas de Evaluación

El dashboard muestra las siguientes métricas:

- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático medio
- **MAE** (Mean Absolute Error): Error absoluto medio
- **MAPE** (Mean Absolute Percentage Error): Porcentaje de error absoluto medio
- **AIC** (Akaike Information Criterion): Criterio de información de Akaike
- **BIC** (Bayesian Information Criterion): Criterio de información Bayesiano

## 💡 Ejemplos de Uso

### 1. Análisis Exploratorio
```python
# Selecciona un activo (ej. AAPL)
# Elige un período de 3 años
# Haz clic en "Cargar Datos"
# Revisa las gráficas ACF/PACF para identificar parámetros
```

### 2. Comparar Modelos
```python
# Ve a "Comparativa de Modelos"
# Selecciona valores de p: [1, 2]
# Selecciona valores de d: [1]
# Selecciona valores de q: [1, 2]
# Ejecuta la comparativa y analiza los resultados
```

### 3. Laboratorio Interactivo
```python
# Prueba con diferentes activos (GOOGL, TSLA, ^GSPC)
# Ajusta parámetros ARIMA libremente
# Modifica el porcentaje de entrenamiento
# Descarga los pronósticos en CSV
```

## 🛠️ Solución de Problemas

### Error al cargar datos
- Verifica que el símbolo del activo sea correcto
- Asegúrate de tener conexión a internet
- Intenta con otro período de tiempo

### Error al ajustar modelo
- Prueba con diferentes parámetros (p, d, q)
- Asegúrate de tener suficientes datos (>100 observaciones)
- Verifica que los parámetros no sean demasiado altos

### La aplicación no inicia
- Verifica que todas las dependencias estén instaladas
- Asegúrate de que el puerto 8501 no esté en uso
- Intenta reiniciar el entorno virtual

## 📝 Notas Importantes

1. **Datos Financieros**: Los precios de acciones son volátiles y los modelos ARIMA tienen limitaciones para predecir mercados financieros.

2. **Fines Educativos**: Este dashboard está diseñado para fines educativos. No lo uses para tomar decisiones de inversión reales.

3. **Parámetros Óptimos**: No existe un conjunto único de parámetros (p, d, q) que funcione para todos los casos. Debes experimentar y validar.

4. **Estacionariedad**: La mayoría de series financieras requieren diferenciación (d ≥ 1) para ser estacionarias.

## 🤝 Contribuciones

Este proyecto es de código abierto. Si encuentras errores o tienes sugerencias:

1. Reporta issues en el repositorio
2. Sugiere nuevas características
3. Comparte tus mejoras

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Siéntete libre de usarlo, modificarlo y distribuirlo.

## 👨‍💻 Autor

Desarrollado con fines educativos para entender los modelos ARIMA.

## 🙏 Agradecimientos

- **Streamlit** por su excelente framework
- **Yahoo Finance** por proporcionar datos gratuitos
- **Statsmodels** por las herramientas de series temporales
- La comunidad de ciencia de datos por su invaluable contribución

---

**¡Disfruta explorando los modelos ARIMA! 📈**