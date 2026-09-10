#!/bin/bash

source /home/dms1/anaconda3/etc/profile.d/conda.sh
conda activate ct5090

export PYTHONNOUSERSITE=1
python -m streamlit run app.py