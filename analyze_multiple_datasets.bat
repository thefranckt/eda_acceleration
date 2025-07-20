@echo off
echo ========================================
echo    EDA Analysis for Multiple Datasets
echo ========================================

echo.
echo Analyzing Dataset 1: Advertising Sales
python src/eda_report_enhanced.py --input "C:/Users/franc/Desktop/Projets/Kaggle/Advertising_sales/Adertising_Sales/data/raw/Advertising.csv" --outdir Reports_Advertising --target "Sales ($)"

echo.
echo ========================================
echo Add your other datasets below:
echo ========================================

REM echo Analyzing Dataset 2: Your Dataset
REM python src/eda_report_enhanced.py --input "path/to/your/dataset2.csv" --outdir Reports_Dataset2 --target "your_target_column"

REM echo Analyzing Dataset 3: Another Dataset  
REM python src/eda_report_enhanced.py --input "path/to/your/dataset3.csv" --outdir Reports_Dataset3 --target "another_target"

echo.
echo ========================================
echo All analyses completed!
echo ========================================
pause
