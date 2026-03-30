Param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ArgsFromCaller
)
# Native PowerShell may block unsigned local scripts. If that happens, run:
# powershell -ExecutionPolicy Bypass -File .\setup_data.ps1
$PyExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$HasConfig = $ArgsFromCaller -contains "--config"
if (-not $HasConfig) {
    & $PyExe setup_data.py --config configs/default.yaml @ArgsFromCaller
} else {
    & $PyExe setup_data.py @ArgsFromCaller
}
