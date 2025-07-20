# Guide de Configuration pour Différents Datasets

## 🔧 Configuration Rapide

### Étapes pour analyser un nouveau dataset :

1. **Préparez votre fichier CSV** - Assurez-vous qu'il soit propre et bien formaté
2. **Identifiez votre colonne cible** - La variable que vous voulez prédire/analyser  
3. **Choisissez un nom de dossier** - Où sauvegarder les résultats
4. **Lancez l'analyse**

## 📊 Exemples par Domaine

### 🏠 **Immobilier**
```bash
python src/eda_report_enhanced.py \
  --input "data/housing_prices.csv" \
  --outdir Reports_Housing \
  --target "price"
```

### 🛒 **E-commerce**  
```bash
python src/eda_report_enhanced.py \
  --input "data/sales_data.csv" \
  --outdir Reports_Sales \
  --target "revenue"
```

### 👥 **Ressources Humaines**
```bash
python src/eda_report_enhanced.py \
  --input "data/employee_data.csv" \
  --outdir Reports_HR \
  --target "salary"
```

### 🏥 **Médical**
```bash
python src/eda_report_enhanced.py \
  --input "data/patient_data.csv" \
  --outdir Reports_Medical \
  --target "diagnosis"
```

### 📈 **Finance**
```bash
python src/eda_report_enhanced.py \
  --input "data/stock_prices.csv" \
  --outdir Reports_Finance \
  --target "closing_price"
```

### 🎓 **Éducation**
```bash
python src/eda_report_enhanced.py \
  --input "data/student_performance.csv" \
  --outdir Reports_Education \
  --target "final_grade"
```

## 🎯 **Conseils pour la Colonne Cible**

### ✅ **Bonnes pratiques :**
- **Régression** : Variables numériques continues (prix, âge, salaire, score)
- **Classification** : Variables catégorielles (oui/non, type, segment, classe)
- **Sans cible** : Utilisez n'importe quel nom, les analyses descriptives fonctionnent quand même

### 📝 **Exemples de colonnes cibles courantes :**
- `price`, `cost`, `revenue`, `sales`
- `target`, `label`, `class`, `category`  
- `score`, `rating`, `grade`, `performance`
- `outcome`, `result`, `status`, `type`

## 🔄 **Workflow Recommandé**

### 1. **Exploration Initiale**
```bash
# Analyse rapide sans colonne cible spécifique
python src/eda_report_enhanced.py \
  --input "mon_dataset.csv" \
  --outdir Reports_Exploration \
  --target "any_column"
```

### 2. **Analyse Ciblée** 
```bash
# Analyse approfondie avec la vraie colonne cible
python src/eda_report_enhanced.py \
  --input "mon_dataset.csv" \
  --outdir Reports_Final \
  --target "ma_vraie_cible"
```

## 📁 **Organisation des Résultats**

```
eda_acceleration/
├── Reports_Dataset1/
│   ├── enhanced_eda_report.html    # ← Ouvrez ce fichier
│   ├── boxplot_*.png
│   ├── violin_*.png
│   └── autres_visualisations.png
├── Reports_Dataset2/
│   ├── enhanced_eda_report.html
│   └── ...
└── Reports_Dataset3/
    ├── enhanced_eda_report.html
    └── ...
```

## ⚡ **Scripts d'Automatisation**

### Utilisez `run_multiple_datasets.py` :
1. Éditez la liste `DATASETS` dans le fichier
2. Ajoutez vos datasets
3. Lancez : `python run_multiple_datasets.py`

### Ou utilisez `analyze_multiple_datasets.bat` :
1. Modifiez le fichier batch
2. Ajoutez vos commandes
3. Double-cliquez pour exécuter

## 🚨 **Dépannage Commun**

### Fichier non trouvé :
- Vérifiez le chemin complet
- Utilisez des slashes `/` ou double backslashes `\\`
- Mettez le chemin entre guillemets si il contient des espaces

### Colonne cible inexistante :
- L'analyse continue quand même
- Seule la feature importance est ignorée
- Vérifiez l'orthographe exacte de la colonne

### Encodage de caractères :
- Assurez-vous que le CSV est en UTF-8
- Évitez les caractères spéciaux dans les noms de colonnes

## 💡 **Astuces Pro**

1. **Nommage cohérent** : Utilisez des préfixes comme `Reports_ProjectName`
2. **Sauvegarde** : Conservez vos différents rapports pour comparaison
3. **Documentation** : Notez vos observations dans chaque rapport
4. **Partage** : Les fichiers HTML peuvent être envoyés par email ou hébergés
