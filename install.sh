#!/bin/bash

echo "========================================"
echo "   Instalador del Dashboard ARIMA"
echo "========================================"
echo ""

# Verificar si Python esta instalado
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 no esta instalado"
    echo "Por favor instala Python 3.8 o superior"
    exit 1
fi

echo "[1/4] Python detectado correctamente"
python3 --version
echo ""

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "[2/4] Creando entorno virtual..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: No se pudo crear el entorno virtual"
        exit 1
    fi
else
    echo "[2/4] El entorno virtual ya existe"
fi
echo ""

# Activar entorno virtual e instalar dependencias
echo "[3/4] Instalando dependencias..."
source venv/bin/activate
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudieron instalar las dependencias"
    echo "Intenta ejecutar: pip install -r requirements.txt manualmente"
    exit 1
fi
echo ""

echo "[4/4] Instalacion completada exitosamente!"
echo ""
echo "========================================"
echo "   Para ejecutar el dashboard:"
echo "   1. Activa el entorno virtual:"
echo "      source venv/bin/activate"
echo "   2. Ejecuta Streamlit:"
echo "      streamlit run app.py"
echo "========================================"
echo ""
echo "¿Deseas ejecutar el dashboard ahora? (s/n)"
read -r run_now

if [[ $run_now =~ ^[Ss]$ ]]; then
    source venv/bin/activate
    streamlit run app.py
else
    echo "El dashboard esta listo para ejecutarse cuando lo necesites."
fi