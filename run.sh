#!/bin/bash

echo "========================================"
echo "   Iniciando Dashboard ARIMA"
echo "========================================"
echo ""

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "ERROR: El entorno virtual no existe"
    echo "Ejecuta install.sh primero para instalar las dependencias"
    exit 1
fi

# Activar entorno virtual
source venv/bin/activate

# Verificar si streamlit esta instalado
if ! python -c "import streamlit" &> /dev/null; then
    echo "ERROR: Streamlit no esta instalado"
    echo "Ejecuta install.sh primero"
    exit 1
fi

echo "Iniciando Streamlit..."
echo "El dashboard se abrira en tu navegador automaticamente"
echo "URL: http://localhost:8501"
echo ""
echo "Para detener el dashboard, presiona Ctrl+C"
echo ""

streamlit run app.py --server.headless true