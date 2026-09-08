<#
fpl.ps1 - the one entry point.
==============================

Every run of this project needs the same four-line preamble: point FPL_DATA at the
cloned FPL-Core-Insights data dir, point FPL_HISTORY at the vaastav clone, find
Python 3.13 (which is not on PATH as `python` or `py`), and force UTF-8 on stdout
so a script printing a Greek letter does not die under a pipe.

That preamble was being retyped into every invocation. The cost is not typing: an
agent's permission allowlist matches on the WHOLE command string, so every new
one-off - a different script, a different tail length - was a fresh entry and a
fresh prompt. The allowlist had grown to ~90 near-identical entries, none of which
ever matched the next command. One wrapper collapses that to one rule.

NOTE ON ENCODING: this file is deliberately pure ASCII. PowerShell 5.1 reads a .ps1
without a BOM as ANSI, so a UTF-8 em-dash arrives as three cp1252 characters, one of
which is a curly quote - and PowerShell treats curly quotes as string delimiters.
The result is a parse error forty lines from the character that caused it. Keep it
ASCII, or save it with a BOM.

Usage
-----
  .\fpl.ps1 doctor            environment + artifact graph + what to rebuild
  .\fpl.ps1 build             regenerate inputs and priors  (scripts/build_all.py)
  .\fpl.ps1 board             the canonical board           (scripts/gw_board.py)
  .\fpl.ps1 test [--quick]    the regression harness        (scripts/test_all.py)
  .\fpl.ps1 paths             resolved paths                (src/config.py)
  .\fpl.ps1 graph [--check]   the artifact DAG              (src/manifest.py)
  .\fpl.ps1 run <path> [args] any script in the repo, with the environment set
  .\fpl.ps1 py <args>         raw python, with the environment set

`run` and `py` are the escape hatches: they take arbitrary arguments, so one
allowlist rule for this wrapper is exactly as permissive as "may run this project's
Python". That is the intended trade - the alternative was ~90 rules that each
authorised one literal command and expired the moment anything changed. Allowlist
only the fixed subcommands if you want a narrower boundary.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Command = "help",
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- Python 3.13. Overridable, because a different machine will put it elsewhere.
if ($env:FPL_PYTHON) {
    $Py = $env:FPL_PYTHON
} else {
    $Py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
}
if (-not (Test-Path $Py)) {
    Write-Error "Python 3.13 not found at $Py. Set FPL_PYTHON to its full path."
}

# --- Data roots. Only set what is not already set, so an explicit env var wins.
if (-not $env:FPL_DATA) {
    $guess = Join-Path (Split-Path -Parent $Root) "FPL-Core-Insights\data"
    if (Test-Path $guess) { $env:FPL_DATA = $guess }
}
if (-not $env:FPL_HISTORY) {
    $guess = Join-Path (Split-Path -Parent $Root) "Fantasy-Premier-League\data"
    if (Test-Path $guess) { $env:FPL_HISTORY = $guess }
}

# Piped stdout on Windows is cp1252 while the console is UTF-8, so a script printing
# a delta or an arrow passes by hand and dies under a pipe. Force UTF-8 here for the
# same reason scripts/test_all.py forces it on every child process.
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Py {
    param([string[]]$PyArgs)
    Push-Location $Root
    try { & $Py -u @PyArgs }
    finally { Pop-Location }
    exit $LASTEXITCODE
}

switch ($Command.ToLower()) {
    "doctor" { Invoke-Py (@("scripts\doctor.py") + $Rest) }
    "build"  { Invoke-Py (@("scripts\build_all.py") + $Rest) }
    "board"  { Invoke-Py (@("scripts\gw_board.py") + $Rest) }
    "test"   { Invoke-Py (@("scripts\test_all.py") + $Rest) }
    "paths"  { Invoke-Py (@("src\config.py") + $Rest) }
    "graph"  { Invoke-Py (@("src\manifest.py") + $Rest) }
    "run" {
        if (-not $Rest -or $Rest.Count -lt 1) { Write-Error "run needs a script path" }
        Invoke-Py $Rest
    }
    "py" { Invoke-Py $Rest }
    default {
        Write-Output 'fpl.ps1 - commands: doctor build board test paths graph run py'
        Write-Output ''
        Write-Output "  python  : $Py"
        Write-Output "  FPL_DATA: $env:FPL_DATA"
        Write-Output "  FPL_HIST: $env:FPL_HISTORY"
        Write-Output ''
        Write-Output '  .\fpl.ps1 doctor             environment + artifact graph + rebuild plan'
        Write-Output '  .\fpl.ps1 test --quick       regression harness, no pipeline or studies'
        Write-Output '  .\fpl.ps1 run studies\x.py   any script, environment already set'
    }
}
