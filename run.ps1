param (
    [Parameter(Position=0)]
    [ValidateSet("setup", "data", "train", "serve", "dashboard", "test", "docker-build", "docker-run", "clean", "help")]
    [string]$Action = "help"
)

$VENV_DIR = ".venv"
$PYTHON = "$VENV_DIR\Scripts\python.exe"
$PIP = "$VENV_DIR\Scripts\pip.exe"
$STREAMLIT = "$VENV_DIR\Scripts\streamlit.exe"
$PYTEST = "$VENV_DIR\Scripts\pytest.exe"

switch ($Action) {
    "setup" {
        Write-Host ">>> Creating virtual environment in $VENV_DIR..." -ForegroundColor Green
        python -m venv $VENV_DIR
        Write-Host ">>> Upgrading pip..." -ForegroundColor Green
        & $PYTHON -m pip install --upgrade pip
        Write-Host ">>> Installing requirements..." -ForegroundColor Green
        & $PIP install -r requirements.txt
        Write-Host ">>> Setup completed successfully!" -ForegroundColor Green
    }
    
    "data" {
        Write-Host ">>> Running data enrichment and splitting..." -ForegroundColor Green
        & $PYTHON -m src.data
    }
    
    "train" {
        Write-Host ">>> Starting model training with MLflow tracking..." -ForegroundColor Green
        & $PYTHON -m src.pipeline
    }
    
    "serve" {
        Write-Host ">>> Starting FastAPI server locally on http://127.0.0.1:8000..." -ForegroundColor Green
        & $PYTHON -m src.api
    }
    
    "dashboard" {
        Write-Host ">>> Starting Streamlit dashboard..." -ForegroundColor Green
        & $STREAMLIT run src/app.py
    }
    
    "test" {
        Write-Host ">>> Executing unit test suite..." -ForegroundColor Green
        & $PYTHON -m pytest tests/ -v
    }
    
    "docker-build" {
        Write-Host ">>> Building Docker container images..." -ForegroundColor Green
        docker-compose build
    }
    
    "docker-run" {
        Write-Host ">>> Starting API and Dashboard services via Docker Compose..." -ForegroundColor Green
        docker-compose up
    }
    
    "clean" {
        Write-Host ">>> Cleaning data directory, models, logs, and caches..." -ForegroundColor Yellow
        if (Test-Path "data\*.csv") { Remove-Item "data\*.csv" -Force }
        if (Test-Path "models\*.pkl") { Remove-Item "models\*.pkl" -Force }
        if (Test-Path "logs\*.log") { Remove-Item "logs\*.log" -Force }
        if (Test-Path "mlruns") { Remove-Item "mlruns" -Recurse -Force }
        if (Test-Path ".pytest_cache") { Remove-Item ".pytest_cache" -Recurse -Force }
        if (Test-Path "src\__pycache__") { Remove-Item "src\__pycache__" -Recurse -Force }
        Write-Host ">>> Cleaned!" -ForegroundColor Green
    }
    
    "help" {
        Write-Host "=== Fraud Detection Orchestration Helper ===" -ForegroundColor Cyan
        Write-Host "Usage: .\run.ps1 <action>" -ForegroundColor White
        Write-Host ""
        Write-Host "Actions:" -ForegroundColor White
        Write-Host "  setup         - Build virtual environment and install requirements" -ForegroundColor Gray
        Write-Host "  data          - Run data preprocessing, enrichment and stratified split" -ForegroundColor Gray
        Write-Host "  train         - Train the model using ColumnTransformer and MLflow" -ForegroundColor Gray
        Write-Host "  serve         - Run the FastAPI prediction service locally" -ForegroundColor Gray
        Write-Host "  dashboard     - Run the Streamlit threshold tuning dashboard locally" -ForegroundColor Gray
        Write-Host "  test          - Execute tests with pytest" -ForegroundColor Gray
        Write-Host "  docker-build  - Build Docker containers" -ForegroundColor Gray
        Write-Host "  docker-run    - Run FastAPI + Streamlit container composition" -ForegroundColor Gray
        Write-Host "  clean         - Remove cached artifacts, models, logs" -ForegroundColor Gray
    }
}
