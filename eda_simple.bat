@echo off
echo.
echo ==========================================
echo    🚀 GENERATEUR EDA SIMPLIFIE 🚀
echo ==========================================
echo.

REM Vérifier les paramètres
if "%~1"=="" (
    echo ❌ ERREUR: Fichier d'entrée requis
    echo.
    echo 📋 UTILISATION:
    echo    eda_simple.bat FICHIER_DONNEES [VARIABLE_CIBLE] [FICHIER_SORTIE]
    echo.
    echo 💡 EXEMPLES:
    echo    eda_simple.bat donnees.csv
    echo    eda_simple.bat donnees.csv prix_vente
    echo    eda_simple.bat donnees.csv prix_vente mon_rapport.html
    echo.
    pause
    exit /b 1
)

set INPUT_FILE=%~1
set TARGET_VAR=%~2
set OUTPUT_FILE=%~3

REM Vérifier l'existence du fichier d'entrée
if not exist "%INPUT_FILE%" (
    echo ❌ ERREUR: Fichier '%INPUT_FILE%' non trouvé
    pause
    exit /b 1
)

echo 📂 Fichier d'entrée: %INPUT_FILE%
if not "%TARGET_VAR%"=="" (
    echo 🎯 Variable cible: %TARGET_VAR%
)
if not "%OUTPUT_FILE%"=="" (
    echo 📁 Fichier de sortie: %OUTPUT_FILE%
)

echo.
echo 🔄 Génération du rapport EDA...
echo.

REM Construire la commande Python
set PYTHON_CMD=python src\eda_simple.py -i "%INPUT_FILE%"

if not "%TARGET_VAR%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% -t "%TARGET_VAR%"
)

if not "%OUTPUT_FILE%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% -o "%OUTPUT_FILE%"
)

REM Exécuter le script Python
%PYTHON_CMD%

if %errorlevel% equ 0 (
    echo.
    echo ✅ RAPPORT GENERE AVEC SUCCES!
    echo.
    
    REM Déterminer le nom du fichier de sortie
    if not "%OUTPUT_FILE%"=="" (
        set FINAL_OUTPUT=%OUTPUT_FILE%
    ) else (
        REM Extraire le nom de base du fichier d'entrée
        for %%f in ("%INPUT_FILE%") do set BASENAME=%%~nf
        set FINAL_OUTPUT=eda_report_!BASENAME!.html
    )
    
    echo 📁 Fichier généré: !FINAL_OUTPUT!
    echo.
    
    REM Proposer d'ouvrir le rapport
    set /p OPEN_REPORT=🌐 Ouvrir le rapport dans le navigateur? (O/N): 
    if /i "!OPEN_REPORT!"=="O" (
        start "" "!FINAL_OUTPUT!"
    )
    
) else (
    echo.
    echo ❌ ERREUR lors de la génération du rapport
    echo Vérifiez les logs ci-dessus pour plus de détails
)

echo.
pause
