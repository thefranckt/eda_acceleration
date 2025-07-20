@echo off
echo 🚀 TEST ULTRA-SIMPLE 
echo.
echo Génération du rapport avec vos données publicitaires...
python src\eda_simple.py -i advertising_data.csv -t Sales_dollar -o TEST_FINAL.html
echo.
echo ✅ Terminé ! 
echo 📂 Ouvrez le fichier: TEST_FINAL.html
echo.
pause
