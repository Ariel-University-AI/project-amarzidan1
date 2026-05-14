@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install streamlit pandas plotly --quiet
echo.
echo Opening EDA App...
streamlit run eda_app.py
pause
