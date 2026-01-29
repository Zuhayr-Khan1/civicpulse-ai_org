# start_jupyter.ps1
# Jupyter Launcher Script for Windows PowerShell

# Get project directory
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Navigate to project
Set-Location $ProjectDir

# Check if venv exists
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found at .\venv\Scripts\Activate.ps1" -ForegroundColor Red
    Write-Host "Please create it first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Verify Jupyter is installed
Write-Host "✓ Checking Jupyter installation..." -ForegroundColor Green
jupyter --version

# Start Jupyter
Write-Host "`n🚀 Starting Jupyter Notebook..." -ForegroundColor Green
Write-Host "📍 Working directory: $ProjectDir\notebooks" -ForegroundColor Cyan
Write-Host "🌐 Access at: http://localhost:8888" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop Jupyter`n" -ForegroundColor Yellow

# Start Jupyter in notebooks directory
jupyter notebook `
    --notebook-dir="$ProjectDir\notebooks" `
    --ip=127.0.0.1 `
    --port=8888 `
    --no-browser `
    --NotebookApp.iopub_data_rate_limit=10000000

Write-Host "`n✋ Jupyter stopped" -ForegroundColor Yellow
