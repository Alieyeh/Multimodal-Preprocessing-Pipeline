Param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ArgsFromCaller
)
$PyExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$HasConfig = $ArgsFromCaller -contains "--config"
if (-not $HasConfig) {
    & $PyExe preprocess.py --config configs/default.yaml @ArgsFromCaller
} else {
    & $PyExe preprocess.py @ArgsFromCaller
}
