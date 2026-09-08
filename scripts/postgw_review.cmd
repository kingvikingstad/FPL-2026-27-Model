@echo off
REM ---------------------------------------------------------------------------
REM postgw_review.cmd - the after-every-gameweek chore.
REM
REM Registered with Windows Task Scheduler to run DAILY. Gameweeks do not end on
REM a weekday a cron can express, so the trigger is a state test, not a time:
REM --auto reviews the newest gameweek that is complete AND unreviewed, and is a
REM no-op on every other day. Running it daily is therefore correct and cheap.
REM
REM It also refreshes the data clone first - the review is worthless against a
REM stale checkout, and the clone is the only source of realised results.
REM ---------------------------------------------------------------------------
setlocal

set "REPO=%~dp0.."
set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
set "LOG=%REPO%\.cache\postgw_review.log"
set "PYTHONIOENCODING=utf-8"
set "DATA=%REPO%\..\FPL-Core-Insights"

if not exist "%REPO%\.cache" mkdir "%REPO%\.cache"

if not exist "%PY%" (
    >>"%LOG%" echo [%DATE% %TIME%] ERROR python not found at %PY%
    exit /b 1
)

>>"%LOG%" echo [%DATE% %TIME%] refreshing data clone
if exist "%DATA%\.git" git -C "%DATA%" pull --ff-only >>"%LOG%" 2>&1

cd /d "%REPO%" || exit /b 1
>>"%LOG%" echo [%DATE% %TIME%] postgw_review --auto
"%PY%" scripts\postgw_review.py --auto >>"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" >>"%LOG%" echo [%DATE% %TIME%] ERROR exit %RC%
exit /b %RC%
