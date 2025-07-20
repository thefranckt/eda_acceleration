@echo off
chcp 65001 > nul
echo.
echo ========================================
echo   🆕 TEST NOUVELLES DONNÉES 🆕
echo ========================================
echo.

echo 📊 Test 1: Données Employés (CSV virgules)
echo    Analyse du salaire en fonction de l'expérience, âge, formation...
python src\eda_simple.py -i donnees_employes.csv -t salaire -o test_employes.html
echo.

echo 📊 Test 2: Ventes Produits (CSV point-virgule) 
echo    Analyse des prix en fonction des catégories, satisfaction...
python src\eda_simple.py -i ventes_produits.csv -t prix -o test_ventes.html
echo.

echo 📊 Test 3: Résultats Scolaires (TXT avec pipe)
echo    Analyse de la moyenne générale par matière et classe...
python src\eda_simple.py -i resultats_scolaires.txt -t moyenne_generale -o test_scolaire.html
echo.

echo 📊 Test 4: Données Villes (DAT avec espaces)
echo    Analyse de la population avec PIB, chômage, pollution...
python src\eda_simple.py -i donnees_villes.dat -t population -o test_villes.html
echo.

echo ✅ Tous les tests terminés !
echo 📂 Fichiers HTML générés:
echo    - test_employes.html     (Analyse RH)
echo    - test_ventes.html       (Analyse Commerce)
echo    - test_scolaire.html     (Analyse Éducation)
echo    - test_villes.html       (Analyse Démographique)
echo.
echo 🌐 Ouvrez ces fichiers dans votre navigateur !
pause
