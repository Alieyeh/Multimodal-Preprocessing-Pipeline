Param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ArgsFromCaller
)
$PyExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $PyExe validate_outputs.py @ArgsFromCaller
