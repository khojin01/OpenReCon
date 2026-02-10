#!/bin/bash

echo "Starting ReCon GUI Application..."
echo "=================================="
echo ""
echo "The application will open in your default browser."
echo "If it doesn't open automatically, navigate to: http://localhost:8501"
echo ""

uv run streamlit run app.py
