# Helper script to install dependencies for the current user (no virtualenv) and run the Streamlit app.
# Usage: Open PowerShell in the project root and run:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\run_no_venv.ps1

Write-Output "Upgrading pip (user site)..."
python -m pip install --upgrade pip --user

Write-Output "Installing requirements into user site-packages... (this may take a few minutes)"
python -m pip install --user -r requirements.txt

Write-Output "Launching Streamlit app using the module runner..."
python -m streamlit run .\streamlit_app.py
