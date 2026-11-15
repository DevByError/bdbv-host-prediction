#!/bin/bash

echo ""
echo "================================================================================"
echo "BDBV HOST PREDICTION - WEB APPLICATION LAUNCHER"
echo "================================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "[1] Installing web dependencies..."
pip install -r requirements-web.txt

echo ""
echo "[2] Starting Flask web server..."
echo ""
echo "================================================================================"
echo "WEB APP IS STARTING..."
echo "================================================================================"
echo ""
echo "Open your browser and navigate to:"
echo "  http://localhost:5000"
echo ""
echo "The terminal will show server logs below."
echo "Press Ctrl+C to stop the server."
echo ""
echo "================================================================================"
echo ""

python3 app.py
