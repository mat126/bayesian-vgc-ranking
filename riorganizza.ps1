# riorganizza.ps1
# Riorganizza la cartella piatta del progetto nella struttura definitiva per git.
# Lancialo UNA VOLTA dalla radice del progetto (C:\Users\mmore\Desktop\ranking_vgc):
#     powershell -ExecutionPolicy Bypass -File riorganizza.ps1
# Non cancella nulla: sposta soltanto. I file gia' al posto giusto vengono ignorati.

Write-Host "Riorganizzazione progetto ranking_vgc..." -ForegroundColor Cyan

# 1. crea le cartelle
foreach ($d in @("src", "notebooks", "docs", "results", "models", "figures")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# 2. codice -> src/
$code = @(
    "limitless_vgc.py", "pokepaste.py", "inspect_standings.py", "showdown_replays.py",
    "canonicalize.py", "design_matrix.py", "bt_bayes.py",
    "neural_bt.py", "neural_data.py", "neural_train.py", "neural_eval.py",
    "controlli_dati.py"
)
foreach ($f in $code) { if (Test-Path $f) { Move-Item $f "src\" -Force } }

# 3. notebook -> notebooks/
if (Test-Path "analisi_vgc.ipynb") { Move-Item "analisi_vgc.ipynb" "notebooks\" -Force }

# 4. diagramma -> docs/
if (Test-Path "architettura_neural_bt.svg") { Move-Item "architettura_neural_bt.svg" "docs\" -Force }

# 5. CSV dei ranking -> results/
Get-ChildItem -Filter "ranking_*.csv" -File -ErrorAction SilentlyContinue |
    ForEach-Object { Move-Item $_.FullName "results\" -Force }

# 6. artefatti pesanti -> models/
foreach ($f in @("bt_posterior.nc", "bt_artifacts.json", "neural_ckpt.pt", "neural_vocab.json")) {
    if (Test-Path $f) { Move-Item $f "models\" -Force }
}

# 7. pulizia file di scarto
foreach ($f in @("tempCodeRunnerFile.py", "struttura.txt")) {
    if (Test-Path $f) { Remove-Item $f -Force }
}
if (Test-Path "__pycache__") { Remove-Item "__pycache__" -Recurse -Force }

Write-Host "`nFatto. Struttura risultante:" -ForegroundColor Green
Get-ChildItem -Directory | Select-Object Name
Write-Host "`nProssimi passi:" -ForegroundColor Yellow
Write-Host "  git init"
Write-Host "  git add ."
Write-Host "  git commit -m 'Modello Bradley-Terry gerarchico bayesiano per ranking VGC'"
Write-Host "`nVerifica che la pipeline giri dalla radice, ad esempio:"
Write-Host "  python src/design_matrix.py data/matches_clean.jsonl"
