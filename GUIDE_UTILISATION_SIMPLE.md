# 🚀 Script EDA Simplifié - Guide d'Utilisation

## 📋 Description

Le script `eda_simple.py` génère un rapport d'analyse exploratoire de données (EDA) complet dans un seul fichier HTML avec toutes les visualisations intégrées.

## ✨ Avantages

- ✅ **Une seule commande** pour générer un rapport complet
- ✅ **Un seul fichier HTML** en sortie (visualisations intégrées)
- ✅ **Détection automatique** du format de fichier (CSV, TXT, TSV, DAT)
- ✅ **Support universel** des séparateurs (`,` `;` `|` `\t` espace)
- ✅ **Interface moderne** responsive et professionnelle
- ✅ **Navigation fluide** entre les sections

## 🎯 Utilisation Basique

### Syntaxe générale
```bash
python src/eda_simple.py -i FICHIER_DONNEES [-t VARIABLE_CIBLE] [-o RAPPORT_HTML]
```

### Exemples d'utilisation

#### 1. Analyse simple (sans variable cible)
```bash
python src/eda_simple.py -i mes_donnees.csv
```
**Sortie** : `eda_report_mes_donnees.html`

#### 2. Analyse avec variable cible
```bash
python src/eda_simple.py -i mes_donnees.csv -t prix_vente
```
**Bonus** : Ajoute l'analyse d'importance des variables

#### 3. Nom de fichier personnalisé
```bash
python src/eda_simple.py -i mes_donnees.csv -t prix_vente -o mon_rapport.html
```

## 📊 Contenu du Rapport HTML

### 🎨 **Sections du rapport :**

1. **📈 Aperçu Général**
   - Statistiques globales (lignes, colonnes, valeurs manquantes)
   - Informations sur la variable cible

2. **📊 Distributions des Variables**
   - Histogrammes avec courbes de densité
   - Toutes les variables numériques

3. **📦 Analyse des Outliers (Boxplots)**
   - Détection automatique des valeurs aberrantes
   - Statistiques des quartiles
   - Pourcentages d'outliers

4. **🔗 Matrice de Corrélation**
   - Relations entre variables numériques
   - Coefficients de Pearson
   - Heatmap colorée

5. **🏷️ Variables Catégorielles**
   - Distribution des fréquences
   - Barplots pour chaque variable

6. **⭐ Importance des Variables** *(si variable cible fournie)*
   - Classification ou régression automatique
   - Random Forest feature importance
   - Top 10 des variables les plus importantes

7. **📋 Résumé Détaillé**
   - Tableau récapitulatif complet
   - Statistiques pour chaque variable
   - Types, valeurs manquantes, uniques

## 🔧 Paramètres Détaillés

| Paramètre | Court | Obligatoire | Description |
|-----------|-------|-------------|-------------|
| `--input` | `-i` | ✅ | Chemin vers le fichier de données |
| `--target` | `-t` | ❌ | Variable cible pour l'importance |
| `--output` | `-o` | ❌ | Nom du fichier HTML de sortie |

## 📁 Formats Supportés

### Types de fichiers
- **CSV** : `.csv` (virgules, points-virgules)
- **TXT** : `.txt` (tous séparateurs)
- **TSV** : `.tsv` (tabulations)
- **DAT** : `.dat` (espaces multiples)

### Séparateurs détectés automatiquement
- `,` (virgule)
- `;` (point-virgule) 
- `\t` (tabulation)
- `|` (pipe)
- ` ` (espaces multiples)

### Encodages supportés
- UTF-8, ISO-8859-1, CP1252 (détection automatique)

## 💡 Exemples Pratiques

### Données de vente
```bash
python src/eda_simple.py -i ventes_2024.csv -t chiffre_affaires -o rapport_ventes.html
```

### Données immobilières  
```bash
python src/eda_simple.py -i logements.txt -t prix_m2
```

### Données avec séparateur spécial
```bash
python src/eda_simple.py -i donnees_bancaires.dat -t risque_defaut
```

## 🎨 Fonctionnalités du Rapport HTML

### Design et Navigation
- **Responsive** : s'adapte à tous les écrans
- **Navigation smooth** : liens de navigation fluides
- **Thème moderne** : couleurs professionnelles
- **Sections organisées** : structure claire et logique

### Visualisations Intégrées
- **Images base64** : pas de fichiers externes
- **Haute qualité** : 150 DPI pour l'impression
- **Interactives** : survol et navigation
- **Adaptatives** : taille automatique

### Statistiques Complètes
- **Tableau récapitulatif** : toutes les variables
- **Métriques avancées** : moyennes, médianes, écart-types
- **Détection outliers** : comptages et pourcentages
- **Informations qualité** : valeurs manquantes, uniques

## ⚡ Performance

- **Rapide** : génération en quelques secondes
- **Optimisé** : gestion mémoire efficace
- **Robuste** : gestion d'erreurs complète
- **Léger** : fichier HTML < 1MB typiquement

## 🔍 Dépannage

### Erreur "Variable cible non trouvée"
```bash
# Vérifier les colonnes disponibles
python src/eda_simple.py -i mes_donnees.csv
# Le script affiche les colonnes disponibles dans les logs
```

### Problème de séparateur
Le script détecte automatiquement, mais vous pouvez vérifier :
```bash
# Le script affiche le séparateur détecté dans les logs
2025-07-20 17:54:34,810 - INFO - Detected separator: ','
```

### Fichier vide ou illisible
- Vérifiez l'encodage du fichier
- Assurez-vous qu'il contient des données
- Vérifiez les permissions de lecture

## 🚀 Workflow Recommandé

1. **Première analyse** : Sans variable cible
   ```bash
   python src/eda_simple.py -i donnees.csv
   ```

2. **Identifier** la variable d'intérêt dans le rapport

3. **Analyse approfondie** : Avec variable cible
   ```bash
   python src/eda_simple.py -i donnees.csv -t ma_variable_cible
   ```

4. **Ouvrir** le fichier HTML dans un navigateur

5. **Naviguer** entre les sections pour explorer les données

## ✅ Avantages vs Script Original

| Aspect | Script Original | Script Simplifié |
|--------|----------------|-----------------|
| **Commande** | Multiple fichiers | Une seule commande |
| **Sortie** | Dossier + images | Un seul fichier HTML |
| **Configuration** | Manuelle | Automatique |
| **Portabilité** | Difficile | Excellente |
| **Partage** | Complexe | Un seul fichier |
| **Navigation** | Fichiers séparés | Interface intégrée |

Le script simplifié transforme complètement l'expérience utilisateur en offrant une solution tout-en-un, moderne et facile à utiliser ! 🎉
