Write-Host "Stopping dashboard..." -ForegroundColor Cyan

# Find Vite node processes running from the dashboard directory
$processes = Get-WmiObject Win32_Process -Filter "Name='node.exe'" | Where-Object {
    $_.CommandLine -match "vite"
}

if ($processes.Count -eq 0) {
    Write-Host "No running Vite dashboard found." -ForegroundColor Yellow
    exit 0
}

$processes | ForEach-Object {
    $pid = $_.ProcessId
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped Vite process (PID: $pid)" -ForegroundColor Green
}

Write-Host "Dashboard stopped." -ForegroundColor Cyan
