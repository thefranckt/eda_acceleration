@echo off
chcp 65001 > nul
echo.
echo ========================================
echo   🚀 TEST RAPIDE EDA SIMPLE 🚀
echo ========================================
echo.

echo 📊 Test 1: Données Publicitaires (CSV standard)
python src\eda_simple.py -i advertising_data.csv -t Sales_dollar -o test_advertising_eda.html
echo.

echo 📊 Test 2: Données avec point-virgule
python src\eda_simple.py -i test_data_semicolon.csv -t target -o test_semicolon_eda.html
echo.

echo 📊 Test 3: Données avec pipe
python src\eda_simple.py -i test_data_pipe.txt -t target -o test_pipe_eda.html
echo.

echo ✅ Tests terminés ! Vérifiez les fichiers HTML générés.
echo 📂 Fichiers créés:
echo    - test_advertising_eda.html
echo    - test_semicolon_eda.html  
echo    - test_pipe_eda.html
echo.
pause
