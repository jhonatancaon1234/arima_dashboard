@echo off
echo ========================================
echo   Iniciando Dashboard ARIMA
echo ========================================
echo.

REM Verificar si el entorno virtual existe
if not exist "venv" (
    echo ERROR: El entorno virtual no existe
    echo Ejecuta install.bat primero para instalar las dependencias
    pause
    exit /b 1
)

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Verificar si streamlit esta instalado
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit no esta instalado
    echo Ejecuta install.bat primero
    pause
    exit /b 1
)

echo Iniciando Streamlit...
echo El dashboard se abrira en tu navegador automaticamente
echo URL: http://localhost:8501
echo.
echo Para detener el dashboard, presiona Ctrl+C
echo.

streamlit run app.py --server.headless true

pause