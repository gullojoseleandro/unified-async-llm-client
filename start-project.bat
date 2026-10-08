@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 bootstrap.py %*
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python is required to start the project initializer.
        echo Install Python and try again.
        pause
        exit /b 1
    )
    python bootstrap.py %*
)
set "BOOTSTRAP_EXIT=%ERRORLEVEL%"
if not "%BOOTSTRAP_EXIT%"=="0" (
    echo.
    echo Setup or launch failed. Read the message above for next steps.
    pause
)
exit /b %BOOTSTRAP_EXIT%
