@echo off
REM ============================================
REM Quick Start Script for Streamlit App (Windows)
REM ============================================

echo 🚀 Mental Health Risk Assessment System - Quick Start
echo ======================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+
    pause
    exit /b 1
)

echo ✓ Python found: 
python --version
echo.

REM Check if model exists
if not exist "models\multilabel_xgboost_classifier_chain.pkl" (
    echo ❌ Model file not found: models\multilabel_xgboost_classifier_chain.pkl
    echo Please ensure the model has been trained first.
    pause
    exit /b 1
)

echo ✓ Model file found
echo.

REM Install/upgrade requirements
echo 📦 Installing dependencies...
pip install -q -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed successfully
echo.

echo 🎯 Starting Streamlit app...
echo 📊 App will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

REM Run Streamlit app
streamlit run app\streamlit_app.py --server.port 8501 --server.address localhost

pause
