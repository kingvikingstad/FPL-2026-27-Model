@echo off
REM ---------------------------------------------------------------------------
REM fetch_solio_snapshot.cmd - store one Solio feed snapshot, and one bookmaker
REM odds snapshot (football-data.co.uk, added 2026-09-16). No model change.
REM
REM Registered with Windows Task Scheduler to run every 4 hours, matching the
REM feed's refresh cadence. The Solio endpoint publishes `latest` only and keeps
REM no history, so a refresh not captured is a movement observation that cannot
REM be recovered later - which is the whole reason this runs on a timer.
REM
REM solio_market.fetch() deduplicates on `generatedAt`, so polling more often
REM than the feed refreshes is harmless: it stores nothing and logs "already
REM have". Missing a window is the only failure mode that costs anything.
REM
REM Paths derive from this script's own location (%~dp0 is scripts\), so the
REM repo can move without editing anything here. src\ is the working directory
REM because the engine modules are flat and import each other by name.
REM ---------------------------------------------------------------------------
setlocal

set "REPO=%~dp0.."
set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
set "LOG=%REPO%\.cache\solio_snapshot.log"
REM Piped stdout is cp1252 on Windows; the feed and this repo are UTF-8.
set "PYTHONIOENCODING=utf-8"

if not exist "%REPO%\.cache" mkdir "%REPO%\.cache"

if not exist "%PY%" (
    >>"%LOG%" echo [%DATE% %TIME%] ERROR python not found at %PY%
    exit /b 1
)

cd /d "%REPO%\src" || (
    >>"%LOG%" echo [%DATE% %TIME%] ERROR cannot cd to %REPO%\src
    exit /b 1
)

>>"%LOG%" echo [%DATE% %TIME%] fetch
"%PY%" solio_market.py --fetch >>"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" >>"%LOG%" echo [%DATE% %TIME%] ERROR exit %RC%

REM Bookmaker odds, the second source for per-fixture market lambda
REM (src/fixture_market.py). football-data.co.uk's fixtures.csv rolls a few days
REM ahead and then drops a fixture, so an unfetched day is a price lost for good,
REM exactly as for Solio. fetch_books() dedupes on content, so a poll that finds
REM nothing new stores nothing. Run whatever Solio did: one source failing must
REM not cost the other its observation. The task's exit code is non-zero if
REM EITHER failed, so Task Scheduler's Last Result cannot hide a dead books feed
REM behind a healthy Solio one.
>>"%LOG%" echo [%DATE% %TIME%] fetch books
"%PY%" fixture_market.py --fetch >>"%LOG%" 2>&1
set "RC_BOOKS=%ERRORLEVEL%"
if not "%RC_BOOKS%"=="0" >>"%LOG%" echo [%DATE% %TIME%] ERROR books exit %RC_BOOKS%

if not "%RC%"=="0" exit /b %RC%
exit /b %RC_BOOKS%
