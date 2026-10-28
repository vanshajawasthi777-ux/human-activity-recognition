@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%src\gesture_demo.py"

if not exist "%PYTHON%" (
    echo ERROR: The project Python environment was not found.
    echo Expected: "%PYTHON%"
    echo Restore or create the project's .venv, then launch again.
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo ERROR: The application script was not found.
    echo Expected: "%SCRIPT%"
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not enter the project folder.
    echo Folder: "%PROJECT_DIR%"
    pause
    exit /b 1
)

"%PYTHON%" "%SCRIPT%"
set "APP_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo Human Sense AI exited with an error. Review the message above.
    pause
)
exit /b %APP_EXIT_CODE%