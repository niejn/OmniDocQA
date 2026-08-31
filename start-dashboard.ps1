param(
    [string]$ProjectDir = "D:\mashibing\RAGAS-FINANCE",
    [string]$PluginRoot = "C:\Users\julien\.claude\plugins\cache\understand-anything\understand-anything\2.9.4"
)

$DashboardDir = "$PluginRoot\packages\dashboard"

if (-not (Test-Path $DashboardDir)) {
    Write-Error "Dashboard directory not found: $DashboardDir"
    exit 1
}
if (-not (Test-Path "$ProjectDir\.ua\knowledge-graph.json")) {
    Write-Error "Knowledge graph not found. Run /understand first."
    exit 1
}

Write-Host "Starting dashboard..." -ForegroundColor Cyan

# Launch Vite in a new window so the token URL stays visible
$env:GRAPH_DIR = $ProjectDir
Start-Process powershell -ArgumentList "-NoExit", "-c", "npx vite --host 127.0.0.1" `
    -WorkingDirectory $DashboardDir `
    -WindowStyle Normal

Write-Host "Dashboard started in a new window." -ForegroundColor Green
Write-Host "Look for the `"Dashboard URL`" line in that window." -ForegroundColor Yellow
Write-Host "To stop: .\stop-dashboard.ps1" -ForegroundColor Gray
