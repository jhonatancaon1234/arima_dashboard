@echo off
echo ========================================
echo   Instalador del Dashboard ARIMA
echo ========================================
echo.

REM Verificar si Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Por favor instala Python 3.8 o superior desde python.org
    pause
    exit /b 1
)

echo [1/4] Python detectado correctamente
python --version
echo.

REM Crear entorno virtual si no existe
if not exist "venv" (
    echo [2/4] Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
) else (
    echo [2/4] El entorno virtual ya existe
)
echo.

REM Activar entorno virtual e instalar dependencias
echo [3/4] Instalando dependencias...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: No se pudieron instalar las dependencias
    echo Intenta ejecutar: pip install -r requirements.txt manualmente
    pause
    exit /b 1
)
echo.

echo [4/4] Instalacion completada exitosamente!
echo.
echo ========================================
echo   Para ejecutar el dashboard:
echo   1. Activa el entorno virtual:
echo      venv\Scripts\activate
echo   2. Ejecuta Streamlit:
echo      streamlit run app.py
echo ========================================
echo.
echo ¿Deseas ejecutar el dashboard ahora? (S/N)
set /p run_now=

if /i "%run_now%"=="S" (
    call venv\Scripts\activate.bat
    streamlit run app.py
) else (
    echo El dashboard esta listo para ejecutarse cuando lo necesites.
)

pause