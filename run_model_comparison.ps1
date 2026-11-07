param(
    [int]$Epochs = 30,
    [int]$Patience = 5,
    [int]$BatchSize = 16,
    [int]$ImageSize = 224
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path (Split-Path $repoRoot -Parent) ".venv312\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Missing Python 3.12 environment: $python"
}

$missing = @()
foreach ($package in @("tensorflow", "pandas", "numpy", "cv2", "sklearn")) {
    & $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$package') else 1)"
    if ($LASTEXITCODE -ne 0) { $missing += $package }
}
if ($missing.Count -gt 0) {
    throw "Missing packages: $($missing -join ', '). Install with: $python -m pip install -r requirements-comparison.txt"
}

$labels = Join-Path $repoRoot "data_mask.csv"
if (-not (Test-Path $labels)) {
    throw "Missing labels file: $labels"
}
$tcgaFolders = @(Get-ChildItem -Path $repoRoot -Directory -Filter "TCGA_*")
if ($tcgaFolders.Count -eq 0) {
    throw "No TCGA_* dataset folders found under $repoRoot. Download the MRI dataset and extract the folders here."
}

Set-Location $repoRoot
& $python (Join-Path $repoRoot "model_comparison.py") `
    --epochs $Epochs `
    --patience $Patience `
    --batch-size $BatchSize `
    --image-size $ImageSize
exit $LASTEXITCODE
