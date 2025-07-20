import argparse
import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Optional
import warnings
import base64
from io import BytesIO

# Install chardet if not available
try:
    import chardet
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
    import chardet

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    plt.style.use('seaborn-v0_8')
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization libraries not available. Plots will be skipped.")
    VISUALIZATION_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Feature importance will be skipped.")
    SKLEARN_AVAILABLE = False

def detect_file_separator_and_encoding(file_path: str):
    """Automatically detect the file separator and encoding."""
    try:
        # Detect encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            encoding_result = chardet.detect(raw_data)
            detected_encoding = encoding_result['encoding'] or 'utf-8'
        
        # Read sample lines
        with open(file_path, 'r', encoding=detected_encoding) as file:
            sample_lines = [file.readline().strip() for _ in range(5)]
            sample_lines = [line for line in sample_lines if line]
        
        if not sample_lines:
            return None, None
        
        # Test separators
        separators = [';', ',', '\t', '|', ' ']
        best_separator = ','
        max_columns = 1
        
        for separator in separators:
            try:
                column_counts = []
                for line in sample_lines[:3]:
                    if separator == ' ':
                        parts = [part.strip() for part in line.split() if part.strip()]
                    else:
                        parts = line.split(separator)
                    column_counts.append(len(parts))
                
                if column_counts and all(count > 1 for count in column_counts):
                    avg_columns = sum(column_counts) / len(column_counts)
                    if avg_columns > max_columns:
                        max_columns = avg_columns
                        best_separator = separator
            except:
                continue
        
        if best_separator == ' ':
            best_separator = r'\s+'
        
        logger.info(f"Detected encoding: {detected_encoding}")
        sep_display = "space/whitespace" if best_separator == r'\s+' else repr(best_separator)
        logger.info(f"Detected separator: {sep_display}")
        
        return best_separator, detected_encoding
        
    except Exception as e:
        logger.error(f"Error detecting file format: {str(e)}")
        return ',', 'utf-8'

def smart_read_csv(file_path: str):
    """Intelligently read CSV/TXT files with automatic format detection."""
    try:
        separator, encoding = detect_file_separator_and_encoding(file_path)
        
        if separator is None:
            return None
        
        # Prepare read parameters
        read_params = {
            'encoding': encoding,
            'sep': separator if separator != r'\s+' else None,
            'engine': 'python' if separator == r'\s+' else 'c'
        }
        
        if separator == r'\s+':
            read_params['delim_whitespace'] = True
            del read_params['sep']
        
        # Try to read with detected parameters
        try:
            df = pd.read_csv(file_path, **read_params)
            
            # Validate result
            if df.shape[1] == 1 and separator != r'\s+':
                # Try alternatives
                for alt_sep in [';', ',', '\t', '|']:
                    if alt_sep != separator:
                        try:
                            df_alt = pd.read_csv(file_path, sep=alt_sep, encoding=encoding)
                            if df_alt.shape[1] > 1:
                                logger.info(f"Used alternative separator: {repr(alt_sep)}")
                                return df_alt
                        except:
                            continue
            
            if df.shape[0] == 0 or df.shape[1] == 0:
                logger.error("No valid data found")
                return None
            
            logger.info(f"✅ File successfully read: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error reading with detected parameters: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Critical error in smart_read_csv: {str(e)}")
        return None

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    plot_data = buffer.getvalue()
    buffer.close()
    return base64.b64encode(plot_data).decode()

def create_distribution_plots(df):
    """Create distribution plots for numerical columns."""
    if not VISUALIZATION_AVAILABLE:
        return ""
    
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return ""
        
        cols_to_plot = numeric_cols[:8]  # Limit to 8 columns
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue', alpha=0.7)
            axes[i].set_title(f'Distribution: {col}', fontweight='bold', fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Distributions des Variables Numériques', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_b64 = plot_to_base64(fig)
        plt.close(fig)
        return plot_b64
        
    except Exception as e:
        logger.error(f"Error creating distribution plots: {str(e)}")
        return ""

def create_boxplots(df):
    """Create boxplots for numerical columns."""
    if not VISUALIZATION_AVAILABLE:
        return ""
    
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return ""
        
        cols_to_plot = numeric_cols[:6]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = sns.color_palette("Set2", len(cols_to_plot))
        
        for i, col in enumerate(cols_to_plot):
            sns.boxplot(data=df, y=col, ax=axes[i], color=colors[i])
            
            # Calculate outlier statistics
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            title = f'{col}\n({outlier_count} outliers, {outlier_pct:.1f}%)'
            axes[i].set_title(title, fontweight='bold', fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Boxplots - Détection des Outliers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_b64 = plot_to_base64(fig)
        plt.close(fig)
        return plot_b64
        
    except Exception as e:
        logger.error(f"Error creating boxplots: {str(e)}")
        return ""

def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    if not VISUALIZATION_AVAILABLE:
        return ""
    
    try:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return ""
        
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8},
            ax=ax
        )
        ax.set_title('Matrice de Corrélation', fontsize=16, fontweight='bold')
        
        plot_b64 = plot_to_base64(fig)
        plt.close(fig)
        return plot_b64
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return ""

def create_feature_importance(df, target_col):
    """Create feature importance plot."""
    if not SKLEARN_AVAILABLE or not VISUALIZATION_AVAILABLE:
        return ""
    
    try:
        if target_col not in df.columns:
            return ""
        
        # Clean data
        df_clean = df.dropna(subset=[target_col])
        if df_clean.empty:
            return ""
        
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Only numeric features
        X_num = X.select_dtypes(include="number").fillna(0)
        if X_num.empty:
            return ""
        
        # Determine task type
        if y.dtype == 'object' or y.nunique() <= 10:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            task_type = "Classification"
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            task_type = "Régression"
        
        rf.fit(X_num, y)
        
        # Plot
        importances = pd.Series(rf.feature_importances_, index=X_num.columns)
        importances = importances.sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot.bar(ax=ax, color='steelblue')
        ax.set_title(f"Importance des Variables pour {target_col} ({task_type})", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel("Score d'Importance")
        ax.set_xlabel("Variables")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plot_b64 = plot_to_base64(fig)
        plt.close(fig)
        return plot_b64
        
    except Exception as e:
        logger.error(f"Error creating feature importance: {str(e)}")
        return ""

def create_categorical_plots(df):
    """Create categorical variable plots."""
    if not VISUALIZATION_AVAILABLE:
        return ""
    
    try:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Filter suitable columns
        valid_cols = []
        for col in cat_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 15:
                valid_cols.append(col)
        
        if len(valid_cols) == 0:
            return ""
        
        cols_to_plot = valid_cols[:4]  # Limit to 4 columns
        n_cols = min(2, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Distribution: {col}', fontweight='bold')
            axes[i].set_xlabel('Fréquence')
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Variables Catégorielles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_b64 = plot_to_base64(fig)
        plt.close(fig)
        return plot_b64
        
    except Exception as e:
        logger.error(f"Error creating categorical plots: {str(e)}")
        return ""

def generate_summary_stats(df):
    """Generate summary statistics table."""
    try:
        stats_data = []
        for col in df.columns:
            col_stats = {
                'Variable': col,
                'Type': str(df[col].dtype),
                'Valeurs': df[col].count(),
                'Manquantes': df[col].isnull().sum(),
                'Manquantes (%)': f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
                'Uniques': df[col].nunique(),
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    'Moyenne': f"{df[col].mean():.2f}" if not df[col].isnull().all() else "N/A",
                    'Médiane': f"{df[col].median():.2f}" if not df[col].isnull().all() else "N/A",
                    'Écart-type': f"{df[col].std():.2f}" if not df[col].isnull().all() else "N/A",
                })
            else:
                col_stats.update({
                    'Moyenne': "N/A",
                    'Médiane': "N/A", 
                    'Écart-type': "N/A",
                })
            
            stats_data.append(col_stats)
        
        return stats_data
        
    except Exception as e:
        logger.error(f"Error generating summary stats: {str(e)}")
        return []

def generate_html_report(df, target_col, input_file):
    """Generate complete HTML report with embedded visualizations."""
    try:
        logger.info("Generating HTML report with embedded visualizations...")
        
        # Generate all plots
        dist_plot = create_distribution_plots(df)
        box_plot = create_boxplots(df)
        corr_plot = create_correlation_heatmap(df)
        feature_plot = create_feature_importance(df, target_col) if target_col else ""
        cat_plot = create_categorical_plots(df)
        
        # Generate summary stats
        summary_stats = generate_summary_stats(df)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport EDA - {os.path.basename(input_file)}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .nav-menu {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid #dee2e6;
        }}
        .nav-menu a {{
            color: #667eea;
            text-decoration: none;
            margin: 0 20px;
            font-weight: 500;
            padding: 10px 15px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .nav-menu a:hover {{
            background-color: #667eea;
            color: white;
        }}
        .section {{
            padding: 40px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 1.8em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2.2em;
            font-weight: 300;
        }}
        .stat-card p {{
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .plot-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px 12px;
            text-align: left;
            font-weight: 500;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        tr:hover {{
            background: #e3f2fd;
        }}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            background: #fff3cd;
            color: #856404;
        }}
        .footer {{
            background: #343a40;
            color: white;
            text-align: center;
            padding: 30px;
        }}
        .no-plot {{
            padding: 40px;
            text-align: center;
            color: #6c757d;
            font-style: italic;
            background: #f8f9fa;
            border-radius: 8px;
            border: 2px dashed #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 Rapport d'Analyse EDA</h1>
            <p>Analyse Exploratoire des Données</p>
            <p><strong>Fichier:</strong> {os.path.basename(input_file)} | <strong>Généré le:</strong> {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}</p>
        </div>

        <!-- Navigation -->
        <div class="nav-menu">
            <a href="#overview">📈 Aperçu</a>
            <a href="#distributions">📊 Distributions</a>
            <a href="#boxplots">📦 Boxplots</a>
            <a href="#correlations">🔗 Corrélations</a>
            <a href="#categorical">🏷️ Catégorielles</a>
            {"<a href='#importance'>⭐ Importance</a>" if feature_plot else ""}
            <a href="#details">📋 Détails</a>
        </div>

        <!-- Overview Section -->
        <div id="overview" class="section">
            <h2>📈 Aperçu Général</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{df.shape[0]:,}</h3>
                    <p>Lignes</p>
                </div>
                <div class="stat-card">
                    <h3>{df.shape[1]}</h3>
                    <p>Colonnes</p>
                </div>
                <div class="stat-card">
                    <h3>{df.isnull().sum().sum():,}</h3>
                    <p>Valeurs Manquantes</p>
                </div>
                <div class="stat-card">
                    <h3>{len(df.select_dtypes(include=['number']).columns)}</h3>
                    <p>Variables Numériques</p>
                </div>
                <div class="stat-card">
                    <h3>{len(df.select_dtypes(include=['object', 'category']).columns)}</h3>
                    <p>Variables Catégorielles</p>
                </div>
            </div>
            
            {f'<div class="alert"><strong>Variable Cible:</strong> {target_col}</div>' if target_col else ''}
        </div>

        <!-- Distributions Section -->
        <div id="distributions" class="section">
            <h2>📊 Distributions des Variables</h2>
            <p>Analyse des distributions pour toutes les variables numériques avec histogrammes et courbes de densité.</p>
            {"<div class='plot-container'><img src='data:image/png;base64," + dist_plot + "' alt='Distributions'></div>" if dist_plot else "<div class='no-plot'>Aucune variable numérique trouvée pour les distributions.</div>"}
        </div>

        <!-- Boxplots Section -->
        <div id="boxplots" class="section">
            <h2>📦 Analyse des Outliers (Boxplots)</h2>
            <p>Détection des valeurs aberrantes et analyse des quartiles pour chaque variable numérique.</p>
            {"<div class='plot-container'><img src='data:image/png;base64," + box_plot + "' alt='Boxplots'></div>" if box_plot else "<div class='no-plot'>Aucune variable numérique trouvée pour les boxplots.</div>"}
        </div>

        <!-- Correlations Section -->
        <div id="correlations" class="section">
            <h2>🔗 Matrice de Corrélation</h2>
            <p>Relations linéaires entre les variables numériques (coefficient de Pearson).</p>
            {"<div class='plot-container'><img src='data:image/png;base64," + corr_plot + "' alt='Correlations'></div>" if corr_plot else "<div class='no-plot'>Pas assez de variables numériques pour calculer les corrélations.</div>"}
        </div>

        <!-- Categorical Section -->
        <div id="categorical" class="section">
            <h2>🏷️ Variables Catégorielles</h2>
            <p>Distribution des fréquences pour les variables catégorielles (max 10 modalités par variable).</p>
            {"<div class='plot-container'><img src='data:image/png;base64," + cat_plot + "' alt='Categorical'></div>" if cat_plot else "<div class='no-plot'>Aucune variable catégorielle adaptée trouvée.</div>"}
        </div>
"""

        # Feature Importance Section (only if target_col provided and plot exists)
        if target_col and feature_plot:
            html_content += f"""
        <!-- Feature Importance Section -->
        <div id="importance" class="section">
            <h2>⭐ Importance des Variables</h2>
            <p>Importance relative des variables numériques pour prédire <strong>{target_col}</strong> (Random Forest).</p>
            <div class='plot-container'>
                <img src='data:image/png;base64,{feature_plot}' alt='Feature Importance'>
            </div>
        </div>
"""

        # Details Section with Summary Table
        html_content += f"""
        <!-- Details Section -->
        <div id="details" class="section">
            <h2>📋 Résumé Détaillé des Variables</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Type</th>
                            <th>Valeurs</th>
                            <th>Manquantes</th>
                            <th>Manquantes (%)</th>
                            <th>Uniques</th>
                            <th>Moyenne</th>
                            <th>Médiane</th>
                            <th>Écart-type</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        # Add summary stats rows
        for stat in summary_stats:
            html_content += f"""
                        <tr>
                            <td><strong>{stat['Variable']}</strong></td>
                            <td>{stat['Type']}</td>
                            <td>{stat['Valeurs']}</td>
                            <td>{stat['Manquantes']}</td>
                            <td>{stat['Manquantes (%)']}</td>
                            <td>{stat['Uniques']}</td>
                            <td>{stat['Moyenne']}</td>
                            <td>{stat['Médiane']}</td>
                            <td>{stat['Écart-type']}</td>
                        </tr>"""

        # Close HTML
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p><strong>Rapport EDA Automatique</strong></p>
            <p>Généré avec Python • pandas • matplotlib • seaborn</p>
        </div>
    </div>

    <script>
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });

        // Add loading animation (optional)
        window.addEventListener('load', function() {
            document.body.style.opacity = '1';
        });
    </script>
</body>
</html>
"""

        return html_content
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        return f"<html><body><h1>Erreur lors de la génération du rapport</h1><p>{str(e)}</p></body></html>"

def main():
    parser = argparse.ArgumentParser(
        description="Générateur de rapport EDA avec visualisations intégrées"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Chemin vers le fichier de données (CSV, TXT, TSV, DAT)"
    )
    parser.add_argument(
        "--target", "-t",
        help="Nom de la variable cible pour l'analyse d'importance (optionnel)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Nom du fichier HTML de sortie (par défaut: eda_report.html)"
    )
    
    args = parser.parse_args()
    
    # Default output filename
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"eda_report_{base_name}.html"
    
    try:
        logger.info("🚀 Démarrage de l'analyse EDA...")
        
        # Validate input file
        if not os.path.exists(args.input):
            logger.error(f"❌ Fichier d'entrée non trouvé: {args.input}")
            sys.exit(1)
        
        logger.info(f"📂 Fichier d'entrée: {args.input}")
        if args.target:
            logger.info(f"🎯 Variable cible: {args.target}")
        
        # Read data
        df = smart_read_csv(args.input)
        if df is None:
            logger.error("❌ Impossible de lire le fichier d'entrée")
            sys.exit(1)
        
        logger.info(f"📊 Dataset chargé: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        logger.info(f"📋 Colonnes: {', '.join(df.columns.tolist())}")
        
        # Validate target column
        if args.target and args.target not in df.columns:
            logger.warning(f"⚠️ Variable cible '{args.target}' non trouvée dans les colonnes")
            logger.info(f"Colonnes disponibles: {', '.join(df.columns.tolist())}")
            args.target = None
        
        # Generate HTML report
        logger.info("📝 Génération du rapport HTML...")
        html_content = generate_html_report(df, args.target, args.input)
        
        # Save HTML file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        file_size = os.path.getsize(args.output) / 1024 / 1024  # MB
        logger.info(f"✅ Rapport généré avec succès!")
        logger.info(f"📁 Fichier de sortie: {args.output} ({file_size:.1f} MB)")
        logger.info(f"🌐 Ouvrir dans un navigateur pour voir le rapport complet")
        
    except Exception as e:
        logger.error(f"❌ Erreur critique: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
