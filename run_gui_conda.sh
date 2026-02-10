#!/bin/bash

echo "Starting ReCon GUI Application (conda pytorch_cuda)..."
echo "=========================================="
echo ""
echo "The application will open in your default browser."
echo "If it doesn't open automatically, navigate to: http://localhost:8501"
echo ""

# Activate conda environment and run streamlit
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_cuda
python -m streamlit run app.py
