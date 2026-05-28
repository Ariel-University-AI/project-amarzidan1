@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Streamlit...
python -m streamlit run eda_app.py --server.port 8501
pause
