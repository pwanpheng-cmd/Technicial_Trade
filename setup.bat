@echo off
cd /d "%~dp0"
set PYTHON=C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo ======================================
echo   TECHSCAN SET - Setup
echo ======================================
echo.
echo Working dir: %CD%
echo Python: %PYTHON%
echo.

echo [1/4] Installing core packages...
"%PYTHON%" -m pip install streamlit yfinance plotly anthropic pandas numpy scipy statsmodels tqdm packaging matplotlib --upgrade
echo.

echo [2/4] Installing pandas_ta...
"%PYTHON%" -m pip install pandas_ta --no-deps
echo.

echo [3/4] Creating numba stub (Python 3.14 fix)...
"%PYTHON%" -c "import site, os; sp=site.getsitepackages()[0]; nd=os.path.join(sp,'numba'); os.makedirs(nd,exist_ok=True); open(os.path.join(nd,'__init__.py'),'w').write('def njit(*a,**k):\n    if len(a)==1 and callable(a[0]): return a[0]\n    return lambda f: f\ndef jit(*a,**k):\n    if len(a)==1 and callable(a[0]): return a[0]\n    return lambda f: f\n'); print('numba stub created at: '+nd)"
echo.

echo [4/4] Creating config...
if not exist .streamlit mkdir .streamlit
(
echo [theme]
echo base = "dark"
echo primaryColor = "#00d4ff"
echo backgroundColor = "#080c14"
echo secondaryBackgroundColor = "#0d1828"
echo textColor = "#c8d8e8"
) > .streamlit\config.toml

echo.
echo ======================================
echo   Done! Run run.bat to start.
echo ======================================
pause
