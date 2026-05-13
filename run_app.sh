#!/bin/bash
# ============================================
# Quick Start Script for Streamlit App
# ============================================

echo "🚀 Mental Health Risk Assessment System - Quick Start"
echo "======================================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python --version)"
echo ""

# Check if model exists
if [ ! -f "models/multilabel_xgboost_classifier_chain.pkl" ]; then
    echo "❌ Model file not found: models/multilabel_xgboost_classifier_chain.pkl"
    echo "Please ensure the model has been trained first."
    exit 1
fi

echo "✓ Model file found"
echo ""

# Install/upgrade requirements
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🎯 Starting Streamlit app..."
echo "📊 App will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Run Streamlit app
streamlit run app/streamlit_app.py --server.port 8501 --server.address localhost
