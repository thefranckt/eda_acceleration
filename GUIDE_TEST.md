# 🧪 GUIDE DE TEST - EDA SIMPLE

## 🚀 Tests Rapides

### Test 1: Données Publicitaires (Le plus simple)
```bash
python src\eda_simple.py -i advertising_data.csv -t Sales_dollar
```
**Résultat attendu:** Fichier HTML avec analyse complète des budgets publicitaires

### Test 2: Test avec fichier séparé par point-virgule
```bash
python src\eda_simple.py -i test_data_semicolon.csv -t target
```

### Test 3: Test avec fichier pipe
```bash
python src\eda_simple.py -i test_data_pipe.txt -t target
```

### Test 4: Test avec espaces
```bash
python src\eda_simple.py -i test_data_space.dat -t target
```

## 📋 Vérifications après Test

1. **Fichier HTML généré** ✅
   - Vérifiez qu'un fichier `.html` a été créé
   - Ouvrez-le dans votre navigateur

2. **Contenu du rapport** ✅
   - Navigation fonctionnelle
   - Graphiques intégrés (pas de fichiers séparés)
   - Statistiques détaillées
   - Analyse des outliers
   - Corrélations

3. **Logs dans la console** ✅
   - Messages de progression
   - Détection automatique du séparateur
   - Aucune erreur critique

## 🎯 Test Complet Recommandé

```bash
# Test le plus complet avec le fichier publicitaire
python src\eda_simple.py -i advertising_data.csv -t Sales_dollar -o mon_rapport_complet.html
```

## 🔍 Dépannage

**Si erreur "Module not found":**
```bash
pip install -r requirements.txt
```

**Si problème d'encodage:**
Le script détecte automatiquement l'encodage

**Si colonnes non détectées:**
Vérifiez le séparateur dans vos données - le script teste automatiquement `,;|\t` et espaces

## ⚡ Script de Test Automatique

Exécutez simplement: `.\test_simple.bat`

Ce script teste automatiquement plusieurs formats de fichiers et génère les rapports correspondants.
