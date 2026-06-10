@echo off
cd /d "%~dp0"
echo Starting... http://localhost:8501
python -m streamlit run eda_app.py --server.port 8501
pause
