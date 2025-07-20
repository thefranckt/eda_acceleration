# 🔧 Guide de Détection Automatique de Format - Script EDA Amélioré

## 📋 Vue d'ensemble

Le script EDA amélioré peut maintenant lire automatiquement **n'importe quel fichier de données tabulaires** avec **détection intelligente du format**.

## 🎯 Formats supportés

### Extensions de fichiers
- ✅ `.csv` - Fichiers Comma-Separated Values
- ✅ `.txt` - Fichiers texte avec séparateurs
- ✅ `.tsv` - Fichiers Tab-Separated Values  
- ✅ `.dat` - Fichiers de données
- ✅ **Toute autre extension** - Le script essaiera de lire le fichier automatiquement

### Séparateurs détectés automatiquement
- ✅ `,` (virgule) - Standard international
- ✅ `;` (point-virgule) - Standard européen/allemand
- ✅ `\t` (tabulation) - Fichiers TSV
- ✅ `|` (pipe) - Bases de données/exports
- ✅ ` ` (espaces multiples) - Formats à largeur fixe
- ✅ **Combinaisons** - Le script teste automatiquement

### Encodages détectés automatiquement
- ✅ `UTF-8` - Standard moderne
- ✅ `ISO-8859-1` - Latin-1
- ✅ `CP1252` - Windows Latin-1
- ✅ **Auto-détection** via la librairie `chardet`

## 🚀 Utilisation

### Syntaxe simple
```bash
python src/eda_report_enhanced.py --input VOTRE_FICHIER --outdir DOSSIER_SORTIE --target COLONNE_CIBLE
```

### Exemples concrets

#### Fichier CSV avec virgules
```bash
python src/eda_report_enhanced.py --input data.csv --outdir Reports_CSV --target sales
```

#### Fichier allemand avec points-virgules  
```bash
python src/eda_report_enhanced.py --input daten.csv --outdir Reports_DE --target umsatz
```

#### Fichier TSV avec tabulations
```bash
python src/eda_report_enhanced.py --input export.tsv --outdir Reports_TSV --target revenue
```

#### Fichier TXT avec pipes
```bash
python src/eda_report_enhanced.py --input database_export.txt --outdir Reports_DB --target profit
```

#### Fichier DAT avec espaces
```bash
python src/eda_report_enhanced.py --input measurements.dat --outdir Reports_DAT --target value
```

## 🧠 Intelligence artificielle de détection

### Processus automatique
1. **Détection d'encodage** avec `chardet`
2. **Analyse d'échantillon** des premières lignes
3. **Test des séparateurs** courants
4. **Validation de consistance** des colonnes
5. **Fallback intelligent** en cas d'échec
6. **Rapport de détection** dans les logs

### Exemples de logs
```
2025-07-20 17:01:21,679 - INFO - File analysis complete:
2025-07-20 17:01:21,680 - INFO -   - Detected encoding: utf-8
2025-07-20 17:01:21,680 - INFO -   - Detected separator: '|'
2025-07-20 17:01:21,680 - INFO -   - Expected columns: ~5
2025-07-20 17:01:21,687 - INFO - ✅ File successfully read: 5 rows × 5 columns
```

## 🔍 Validation et sécurité

### Contrôles automatiques
- ✅ **Validation des extensions** (avec avertissement si non-standard)
- ✅ **Détection de colonnes suspectes** (contenant des séparateurs)
- ✅ **Vérification de cohérence** des données lues
- ✅ **Messages d'erreur explicites** en cas de problème
- ✅ **Logs détaillés** pour le debugging

### Gestion d'erreurs
- ✅ **Fallback en cascade** si la détection automatique échoue
- ✅ **Messages d'aide** pour résoudre les problèmes
- ✅ **Validation des données** après lecture
- ✅ **Arrêt propre** si le fichier est illisible

## 📊 Résultats garantis

Une fois le fichier lu avec succès, vous obtenez :

### Visualisations (18 fichiers)
- 📦 **Boxplots** détaillés avec statistiques d'outliers
- 🎻 **Violin plots** avec analyse de distribution  
- 📊 **Histogrammes** avec courbes KDE
- 🔗 **Heatmap de corrélation** 
- ✅ **Analyse de qualité** des données
- 📈 **Feature importance** (si variable cible définie)

### Rapports interactifs
- 📋 **Rapport HTML principal** avec navigation
- 🍯 **SweetViz** interactif
- 📊 **Pandas Profiling** complet

## 🎯 Cas d'usage typiques

### Données européennes/allemandes
```bash
# Fichiers avec points-virgules et encodage spécial
python src/eda_report_enhanced.py --input generali_case_study.csv --outdir Reports_Generali --target schadenmeldung
```

### Exports de bases de données
```bash  
# Fichiers avec pipes ou tabulations
python src/eda_report_enhanced.py --input db_export.txt --outdir Reports_DB --target target_value
```

### Fichiers de mesures scientifiques
```bash
# Fichiers DAT avec espaces et formats fixes
python src/eda_report_enhanced.py --input experiment.dat --outdir Reports_Lab --target measurement
```

## 🆘 Dépannage

### Si le fichier n'est pas lu correctement
1. **Vérifiez les logs** pour voir la détection automatique
2. **Ouvrez le fichier** dans un éditeur de texte pour voir le format
3. **Contactez-nous** avec un échantillon des premières lignes

### Messages d'erreur courants
- `"No data rows found"` → Fichier vide ou que des en-têtes
- `"No columns found"` → Séparateur non détecté, format inhabituel
- `"Failed to read input file"` → Encodage ou format non supporté

## 🎉 Avantages de l'amélioration

### Avant (version simple)
- ❌ Seulement CSV avec virgules
- ❌ Encodage UTF-8 seulement  
- ❌ Détection manuelle du séparateur
- ❌ Erreurs fréquentes avec fichiers européens

### Après (version intelligente)
- ✅ **Tous formats tabulaires**
- ✅ **Auto-détection complète**
- ✅ **Support international**
- ✅ **Robustesse maximale**
- ✅ **Logs explicites**
- ✅ **Zero configuration**

---

## 💡 Conseil pratique

**Plus besoin de vous préoccuper du format !** 
Lancez simplement :
```bash
python src/eda_report_enhanced.py --input VOTRE_FICHIER --outdir Reports --target VOTRE_CIBLE
```

Le script s'occupe automatiquement de tout le reste ! 🚀
