import argparse
import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Optional
import warnings

# Install chardet if not available
try:
    import chardet
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
    import chardet

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging without emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eda_report.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    logger.warning("ydata-profiling not available. Pandas profiling will be skipped.")
    YDATA_AVAILABLE = False

try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    logger.warning("sweetviz not available. SweetViz reports will be skipped.")
    SWEETVIZ_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import missingno as msno
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization libraries not available. Plots will be skipped.")
    VISUALIZATION_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Feature importance will be skipped.")
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available. Statistical plots will be limited.")
    SCIPY_AVAILABLE = False


def detect_file_separator_and_encoding(file_path: str):
    """
    Automatically detect the file separator, encoding, and validate the file format.
    Supports CSV, TXT files with various separators (,;|\t space).
    """
    import csv
    import chardet
    
    try:
        # Detect encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            encoding_result = chardet.detect(raw_data)
            detected_encoding = encoding_result['encoding'] or 'utf-8'
        
        # Read a sample of the file to detect separator
        with open(file_path, 'r', encoding=detected_encoding) as file:
            sample_lines = [file.readline().strip() for _ in range(5)]
            sample_lines = [line for line in sample_lines if line]  # Remove empty lines
        
        if not sample_lines:
            logger.error("File appears to be empty or unreadable")
            return None, None, None
        
        # Test different separators
        separators = [';', ',', '\t', '|', ' ']
        best_separator = ','
        max_columns = 1
        
        for separator in separators:
            try:
                # Count columns for first few lines
                column_counts = []
                for line in sample_lines[:3]:
                    if separator == ' ':
                        # For space separator, split on multiple spaces
                        parts = [part.strip() for part in line.split() if part.strip()]
                    else:
                        parts = line.split(separator)
                    column_counts.append(len(parts))
                
                # Check if we have consistent column count > 1
                if column_counts and all(count > 1 for count in column_counts):
                    avg_columns = sum(column_counts) / len(column_counts)
                    if avg_columns > max_columns:
                        max_columns = avg_columns
                        best_separator = separator
                        
            except Exception as e:
                continue
        
        # Special handling for space-separated files
        if best_separator == ' ':
            # Use regex pattern for multiple spaces
            best_separator = r'\s+'
        
        logger.info(f"File analysis complete:")
        logger.info(f"  - Detected encoding: {detected_encoding}")
        # Handle the backslash issue in f-string
        sep_display = "space/whitespace" if best_separator == r'\s+' else repr(best_separator)
        logger.info(f"  - Detected separator: {sep_display}")
        logger.info(f"  - Expected columns: ~{int(max_columns)}")
        
        return best_separator, detected_encoding, sample_lines[0] if sample_lines else ""
        
    except Exception as e:
        logger.error(f"Error detecting file format: {str(e)}")
        return ',', 'utf-8', ""


def smart_read_csv(file_path: str):
    """
    Intelligently read CSV/TXT files with automatic format detection.
    """
    try:
        separator, encoding, header_sample = detect_file_separator_and_encoding(file_path)
        
        if separator is None:
            logger.error("Could not detect file format")
            return None
        
        # Try to read with detected parameters
        read_params = {
            'encoding': encoding,
            'sep': separator if separator != r'\s+' else None,
            'engine': 'python' if separator == r'\s+' else 'c'
        }
        
        # Handle whitespace separator
        if separator == r'\s+':
            read_params['delim_whitespace'] = True
            del read_params['sep']
        
        # First attempt with detected parameters
        try:
            df = pd.read_csv(file_path, **read_params)
            
            # Validate the result
            if df.shape[1] == 1 and separator != r'\s+':
                # If we got only one column, the separator might be wrong
                logger.warning(f"Only one column detected with separator {separator}, trying alternatives...")
                
                # Try common alternatives
                alternatives = [';', ',', '\t', '|']
                if separator in alternatives:
                    alternatives.remove(separator)
                
                for alt_sep in alternatives:
                    try:
                        df_alt = pd.read_csv(file_path, sep=alt_sep, encoding=encoding)
                        if df_alt.shape[1] > 1:
                            logger.info(f"Successfully read with alternative separator: {repr(alt_sep)}")
                            return df_alt
                    except:
                        continue
            
            # Check if we have reasonable data
            if df.shape[0] == 0:
                logger.error("No data rows found in file")
                return None
            
            if df.shape[1] == 0:
                logger.error("No columns found in file")
                return None
            
            logger.info(f"✅ File successfully read: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error reading with detected parameters: {str(e)}")
            
            # Fallback attempts
            logger.info("Trying fallback methods...")
            fallback_attempts = [
                {'sep': ';', 'encoding': 'utf-8'},
                {'sep': ',', 'encoding': 'utf-8'},
                {'sep': '\t', 'encoding': 'utf-8'},
                {'sep': '|', 'encoding': 'utf-8'},
                {'delim_whitespace': True, 'encoding': 'utf-8'},
                {'sep': ';', 'encoding': 'iso-8859-1'},
                {'sep': ',', 'encoding': 'iso-8859-1'},
                {'sep': ';', 'encoding': 'cp1252'},
                {'sep': ',', 'encoding': 'cp1252'},
            ]
            
            for attempt in fallback_attempts:
                try:
                    df = pd.read_csv(file_path, **attempt)
                    if df.shape[0] > 0 and df.shape[1] > 1:
                        sep_desc = 'whitespace' if attempt.get('delim_whitespace') else f"'{attempt.get('sep')}'"
                        logger.info(f"✅ Fallback successful with separator {sep_desc} and encoding {attempt.get('encoding')}")
                        return df
                except:
                    continue
            
            logger.error("All fallback attempts failed")
            return None
            
    except Exception as e:
        logger.error(f"Critical error in smart_read_csv: {str(e)}")
        return None


def generate_pandas_profile(input_path: str, output_path: str):
    """Generate pandas profiling report with error handling."""
    if not YDATA_AVAILABLE:
        logger.error("ydata-profiling not available. Cannot generate pandas profile.")
        return False
    
    try:
        logger.info("Starting pandas profiling report generation...")
        df = smart_read_csv(input_path)
        
        if df is None:
            logger.error("Failed to read input file for pandas profiling")
            return False
        
        profile = ProfileReport(
            df,
            title="Pandas Profiling Report",
            explorative=True,
            correlations={"pearson": {"calculate": True}},
            minimal=False,  # Set to True for faster generation
        )
        
        output_file = os.path.join(output_path, "pandas_profiling.html")
        profile.to_file(output_file)
        logger.info(f"Pandas profiling report saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating pandas profile: {str(e)}")
        return False

def generate_sweetviz_report(input_path: str, output_path: str):
    """Generate SweetViz report with error handling."""
    if not SWEETVIZ_AVAILABLE:
        logger.error("sweetviz not available. Cannot generate SweetViz report.")
        return False
    
    try:
        logger.info("Starting SweetViz report generation...")
        df = smart_read_csv(input_path)
        
        if df is None:
            logger.error("Failed to read input file for SweetViz report")
            return False
        
        # Monkey patch for NumPy 2.x compatibility
        import numpy as np
        if not hasattr(np, 'VisibleDeprecationWarning'):
            np.VisibleDeprecationWarning = UserWarning
        
        report = sv.analyze(df)
        output_file = os.path.join(output_path, "sweetviz_report.html")
        
        report.show_html(
            filepath=output_file,
            open_browser=False
        )
        logger.info(f"SweetViz report saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating SweetViz report: {str(e)}")
        return False

def plot_missing_matrix(df: pd.DataFrame, output_path: str):
    """Saves a missing‑value matrix to missing_matrix.png"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping missing matrix plot.")
        return False
    
    try:
        logger.info("Creating missing value matrix...")
        plt.figure(figsize=(12, 6))
        msno.matrix(df)
        plt.tight_layout()
        output_file = os.path.join(output_path, "missing_matrix.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Missing matrix saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating missing matrix: {str(e)}")
        plt.close()
        return False

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """Saves a correlation heatmap to corr_heatmap.png"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping correlation heatmap.")
        return False
    
    try:
        logger.info("Creating correlation heatmap...")
        numeric_df = df.select_dtypes(include="number")
        
        if numeric_df.empty:
            logger.warning("No numeric columns found. Skipping correlation heatmap.")
            return False
            
        corr = numeric_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(output_path, "corr_heatmap.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Correlation heatmap saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        plt.close()
        return False

def plot_feature_importance(df: pd.DataFrame, output_path: str, target_col: str):
    """Generate feature importance plot using RandomForest"""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available. Skipping feature importance plot.")
        return False
    
    try:
        logger.info(f"Creating feature importance plot for target: {target_col}")
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")
            return False
        
        # Clean data
        df_clean = df.dropna(subset=[target_col])
        if df_clean.empty:
            logger.warning("No data left after removing missing target values.")
            return False
            
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Only numeric features
        X_num = X.select_dtypes(include="number").fillna(0)
        if X_num.empty:
            logger.warning("No numeric features available for importance calculation.")
            return False
        
        # Determine if it's classification or regression based on target
        if y.dtype == 'object' or y.nunique() <= 10:
            # Classification
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            task_type = "Classification"
        else:
            # Regression
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            task_type = "Regression"
        
        # Fit RandomForest
        rf.fit(X_num, y)
        
        # Plot feature importance
        importances = pd.Series(rf.feature_importances_, index=X_num.columns)
        importances = importances.sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        importances.plot.bar()
        plt.title(f"Top 20 Feature Importances for {target_col} ({task_type})", fontsize=14, fontweight='bold')
        plt.ylabel("Importance Score")
        plt.xlabel("Features")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_file = os.path.join(output_path, "feature_importance.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        plt.close()
        return False

def plot_distribution_overview(df: pd.DataFrame, output_path: str):
    """Create distribution plots for numerical columns"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping distribution plots.")
        return False
    
    try:
        logger.info("Creating distribution overview...")
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for distribution plots.")
            return False
        
        # Limit to first 12 columns to avoid overcrowding
        cols_to_plot = numeric_cols[:12]
        n_cols = min(4, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_file = os.path.join(output_path, "distribution_overview.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Distribution overview saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating distribution overview: {str(e)}")
        plt.close()
        return False

def plot_categorical_overview(df: pd.DataFrame, output_path: str):
    """Create bar plots for categorical columns"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping categorical plots.")
        return False
    
    try:
        logger.info("Creating categorical overview...")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) == 0:
            logger.warning("No categorical columns found for categorical plots.")
            return False
        
        # Filter columns with reasonable number of unique values
        valid_cols = []
        for col in cat_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:  # Between 2 and 20 unique values
                valid_cols.append(col)
        
        if len(valid_cols) == 0:
            logger.warning("No suitable categorical columns found (need 2-20 unique values).")
            return False
        
        # Limit to first 8 columns
        cols_to_plot = valid_cols[:8]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
            axes[i].set_xlabel('Count')
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_file = os.path.join(output_path, "categorical_overview.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Categorical overview saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating categorical overview: {str(e)}")
        plt.close()
        return False

def plot_boxplot_overview(df: pd.DataFrame, output_path: str):
    """Create comprehensive boxplots for numerical columns with outlier statistics"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping boxplot overview.")
        return False
    
    try:
        logger.info("Creating enhanced boxplot overview with outlier statistics...")
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for boxplots.")
            return False
        
        # Limit to first 8 columns for better visibility
        cols_to_plot = numeric_cols[:8]
        n_cols = min(2, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Color palette for better visibility
        colors = sns.color_palette("Set2", len(cols_to_plot))
        
        for i, col in enumerate(cols_to_plot):
            # Create boxplot with custom styling
            box_plot = sns.boxplot(data=df, y=col, ax=axes[i], color=colors[i], width=0.6)
            
            # Calculate outlier statistics
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            # Enhanced title with outlier info
            title = f'{col}\n({outlier_count} outliers, {outlier_percentage:.1f}%)'
            axes[i].set_title(title, fontweight='bold', fontsize=12)
            
            # Add statistical annotations
            median = df[col].median()
            axes[i].axhline(y=median, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add text box with statistics
            stats_text = f'Median: {median:.2f}\nIQR: {IQR:.2f}\nOutliers: {outlier_count}'
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='white', alpha=0.8), fontsize=9)
            
            # Improve axis labels
            axes[i].set_ylabel(col, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Boxplot Overview - Outlier Detection Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_file = os.path.join(output_path, "boxplot_overview.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Enhanced boxplot overview saved to: {output_file}")
        logger.info(f"Analyzed {len(cols_to_plot)} numerical columns for outliers")
        return True
        
    except Exception as e:
        logger.error(f"Error creating boxplot overview: {str(e)}")
        plt.close()
        return False

def plot_individual_boxplots(df: pd.DataFrame, output_path: str):
    """Create individual detailed boxplots for each numerical column"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping individual boxplots.")
        return False
    
    try:
        logger.info("Creating individual detailed boxplots...")
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for individual boxplots.")
            return False
        
        # Create individual boxplot for each column
        created_files = []
        for col in numeric_cols[:6]:  # Limit to first 6 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Boxplot on the left
            sns.boxplot(data=df, y=col, ax=ax1, color='lightblue')
            
            # Calculate detailed statistics
            Q1 = df[col].quantile(0.25)
            Q2 = df[col].quantile(0.50)  # Median
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            
            # Enhanced boxplot annotations
            ax1.axhline(y=Q2, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Median')
            ax1.set_title(f'Boxplot: {col}', fontweight='bold', fontsize=14)
            ax1.set_ylabel(col, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Distribution histogram on the right
            ax2.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(Q2, color='red', linestyle='--', linewidth=2, label='Median')
            ax2.axvline(df[col].mean(), color='orange', linestyle='--', linewidth=2, label='Mean')
            ax2.set_title(f'Distribution: {col}', fontweight='bold', fontsize=14)
            ax2.set_xlabel(col, fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add statistics text box
            stats_text = f"""Statistics for {col}:
• Count: {df[col].count():,}
• Mean: {df[col].mean():.2f}
• Median: {Q2:.2f}
• Std Dev: {df[col].std():.2f}
• Q1: {Q1:.2f}
• Q3: {Q3:.2f}
• IQR: {IQR:.2f}
• Outliers: {outlier_count} ({(outlier_count/len(df)*100):.1f}%)
• Missing: {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df)*100):.1f}%)"""
            
            fig.text(0.02, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            plt.suptitle(f'Detailed Analysis: {col}', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0.15, 0.03, 1, 0.95])
            
            # Save individual file
            safe_col_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('$', 'dollar')
            output_file = os.path.join(output_path, f"boxplot_detail_{safe_col_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            created_files.append(output_file)
            logger.info(f"Individual boxplot saved: {output_file}")
        
        logger.info(f"Created {len(created_files)} individual boxplot files")
        return True
        
    except Exception as e:
        logger.error(f"Error creating individual boxplots: {str(e)}")
        plt.close()
        return False

def plot_data_quality_summary(df: pd.DataFrame, output_path: str):
    """Create a comprehensive data quality summary"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping data quality summary.")
        return False
    
    try:
        logger.info("Creating data quality summary...")
        
        # Collect data quality metrics
        quality_metrics = []
        for col in df.columns:
            metrics = {
                'Column': col,
                'Data_Type': str(df[col].dtype),
                'Missing_Count': df[col].isnull().sum(),
                'Missing_Percentage': (df[col].isnull().sum() / len(df)) * 100,
                'Unique_Count': df[col].nunique(),
                'Unique_Percentage': (df[col].nunique() / len(df)) * 100
            }
            quality_metrics.append(metrics)
        
        quality_df = pd.DataFrame(quality_metrics)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Missing values
        missing_data = quality_df[quality_df['Missing_Count'] > 0]
        if not missing_data.empty:
            sns.barplot(data=missing_data, x='Missing_Percentage', y='Column', ax=ax1)
            ax1.set_title('Missing Data by Column (%)', fontweight='bold')
            ax1.set_xlabel('Missing Percentage (%)')
        else:
            ax1.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            ax1.set_title('Missing Data by Column (%)', fontweight='bold')
        
        # Data types distribution
        type_counts = quality_df['Data_Type'].value_counts()
        ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax2.set_title('Data Types Distribution', fontweight='bold')
        
        # Unique values count
        sns.barplot(data=quality_df.head(10), x='Unique_Count', y='Column', ax=ax3)
        ax3.set_title('Unique Values Count (Top 10)', fontweight='bold')
        ax3.set_xlabel('Number of Unique Values')
        
        # Dataset overview stats
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        stats_text = f"""Dataset Overview:
        
Rows: {len(df):,}
Columns: {len(df.columns)}
Total Cells: {total_cells:,}
Missing Cells: {missing_cells:,} ({(missing_cells/total_cells)*100:.1f}%)
Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"""
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12, verticalalignment='center')
        ax4.set_title('Dataset Statistics', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(output_path, "data_quality_summary.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Data quality summary saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating data quality summary: {str(e)}")
        plt.close()
        return False

def plot_violin_overview(df: pd.DataFrame, output_path: str):
    """Create comprehensive violin plots for numerical columns with distribution details"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping violin plots.")
        return False
    
    try:
        logger.info("Creating enhanced violin plot overview with distribution analysis...")
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for violin plots.")
            return False
        
        # Limit to first 8 columns for better visibility
        cols_to_plot = numeric_cols[:8]
        n_cols = min(2, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Color palette for better visibility
        colors = sns.color_palette("husl", len(cols_to_plot))
        
        for i, col in enumerate(cols_to_plot):
            # Create violin plot with enhanced styling
            violin_plot = sns.violinplot(data=df, y=col, ax=axes[i], color=colors[i], inner="box")
            
            # Calculate distribution statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            skewness = df[col].skew()
            
            # Determine skewness description
            if abs(skewness) < 0.5:
                skew_desc = "Symmetric"
            elif skewness > 0.5:
                skew_desc = "Right-skewed"
            else:
                skew_desc = "Left-skewed"
            
            # Enhanced title with distribution info
            title = f'{col}\n{skew_desc} (skew: {skewness:.2f})'
            axes[i].set_title(title, fontweight='bold', fontsize=12)
            
            # Add mean and median lines
            axes[i].axhline(y=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Mean')
            axes[i].axhline(y=median_val, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Median')
            
            # Add text box with distribution statistics
            stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}\nSkewness: {skewness:.2f}'
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='lightyellow', alpha=0.9), fontsize=10)
            
            # Improve axis labels
            axes[i].set_ylabel(col, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Violin Plot Overview - Distribution Shape Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_file = os.path.join(output_path, "violin_overview.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Enhanced violin plot overview saved to: {output_file}")
        logger.info(f"Analyzed {len(cols_to_plot)} numerical columns for distribution shapes")
        return True
        
    except Exception as e:
        logger.error(f"Error creating violin plot overview: {str(e)}")
        plt.close()
        return False

def plot_individual_violin_plots(df: pd.DataFrame, output_path: str):
    """Create individual detailed violin plots for each numerical column"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping individual violin plots.")
        return False
    
    try:
        logger.info("Creating individual detailed violin plots...")
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for individual violin plots.")
            return False
        
        # Create individual violin plot for each column
        created_files = []
        for col in numeric_cols[:6]:  # Limit to first 6 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Violin plot on the left with quartile information
            sns.violinplot(data=df, y=col, ax=ax1, color='lightcoral', inner="quart")
            
            # Calculate detailed statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            std_val = df[col].std()
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            
            # Enhanced violin plot annotations
            ax1.axhline(y=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Mean')
            ax1.axhline(y=median_val, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Median')
            ax1.axhline(y=Q1, color='green', linestyle=':', alpha=0.6, linewidth=1.5, label='Q1')
            ax1.axhline(y=Q3, color='green', linestyle=':', alpha=0.6, linewidth=1.5, label='Q3')
            
            ax1.set_title(f'Violin Plot: {col}', fontweight='bold', fontsize=14)
            ax1.set_ylabel(col, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            
            # Density plot on the right
            try:
                from scipy.stats import gaussian_kde
                data_clean = df[col].dropna()
                if len(data_clean) > 1:
                    # Create density plot
                    density = gaussian_kde(data_clean)
                    xs = np.linspace(data_clean.min(), data_clean.max(), 200)
                    density_vals = density(xs)
                    
                    ax2.fill_between(xs, density_vals, alpha=0.7, color='skyblue', label='Density')
                    ax2.axvline(mean_val, color='red', linestyle='-', linewidth=2, label='Mean')
                    ax2.axvline(median_val, color='blue', linestyle='--', linewidth=2, label='Median')
                    
                    ax2.set_title(f'Density Curve: {col}', fontweight='bold', fontsize=14)
                    ax2.set_xlabel(col, fontweight='bold')
                    ax2.set_ylabel('Density', fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'Insufficient data for density plot', 
                            ha='center', va='center', transform=ax2.transAxes)
            except ImportError:
                # Fallback to histogram if scipy not available
                ax2.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                ax2.axvline(mean_val, color='red', linestyle='-', linewidth=2, label='Mean')
                ax2.axvline(median_val, color='blue', linestyle='--', linewidth=2, label='Median')
                ax2.set_title(f'Histogram: {col}', fontweight='bold', fontsize=14)
                ax2.set_xlabel(col, fontweight='bold')
                ax2.set_ylabel('Density', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            # Add comprehensive statistics text box
            stats_text = f"""Distribution Statistics for {col}:
• Count: {df[col].count():,}
• Mean: {mean_val:.3f}
• Median: {median_val:.3f}
• Std Dev: {std_val:.3f}
• Min: {df[col].min():.3f}
• Max: {df[col].max():.3f}
• Q1: {Q1:.3f}
• Q3: {Q3:.3f}
• Skewness: {skewness:.3f}
• Kurtosis: {kurtosis:.3f}
• Missing: {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df)*100):.1f}%)"""
            
            # Interpretation of skewness and kurtosis
            if abs(skewness) < 0.5:
                skew_interp = "Approximately symmetric"
            elif skewness > 0.5:
                skew_interp = "Right-skewed (tail extends right)"
            else:
                skew_interp = "Left-skewed (tail extends left)"
            
            if abs(kurtosis) < 0.5:
                kurt_interp = "Normal tail thickness"
            elif kurtosis > 0.5:
                kurt_interp = "Heavy tails (outlier-prone)"
            else:
                kurt_interp = "Light tails (few outliers)"
            
            interp_text = f"""
Distribution Shape:
• {skew_interp}
• {kurt_interp}"""
            
            full_stats = stats_text + interp_text
            
            fig.text(0.02, 0.5, full_stats, fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            
            plt.suptitle(f'Detailed Violin Analysis: {col}', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0.18, 0.03, 1, 0.95])
            
            # Save individual file
            safe_col_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('$', 'dollar')
            output_file = os.path.join(output_path, f"violin_detail_{safe_col_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            created_files.append(output_file)
            logger.info(f"Individual violin plot saved: {output_file}")
        
        logger.info(f"Created {len(created_files)} individual violin plot files")
        return True
        
    except Exception as e:
        logger.error(f"Error creating individual violin plots: {str(e)}")
        plt.close()
        return False

def generate_custom_html_report(df: pd.DataFrame, output_path: str, target_col: str = None):
    """Generate a custom HTML report with embedded visualizations"""
    try:
        logger.info("Creating custom HTML report with embedded visualizations...")
        
        # HTML template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - Enhanced Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .stat-card p {{
            margin: 0;
            font-size: 1.1em;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-container {{
            text-align: center;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .image-container h3 {{
            margin: 15px 0 5px 0;
            color: #495057;
        }}
        .image-container p {{
            margin: 5px 0;
            color: #6c757d;
            font-size: 0.9em;
        }}
        .boxplot-detail {{
            margin: 20px 0;
            padding: 15px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
        }}
        .nav-menu {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .nav-menu a {{
            color: #667eea;
            text-decoration: none;
            margin: 0 15px;
            font-weight: 500;
        }}
        .nav-menu a:hover {{
            text-decoration: underline;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .table th {{
            background: #667eea;
            color: white;
        }}
        .table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Enhanced EDA Report</h1>
            <p>Comprehensive Exploratory Data Analysis with Advanced Visualizations</p>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="nav-menu">
            <a href="#overview">📈 Overview</a>
            <a href="#boxplots">📦 Boxplot Analysis</a>
            <a href="#violinplots">🎻 Violin Plot Analysis</a>
            <a href="#distributions">📊 Distributions</a>
            <a href="#correlations">🔗 Correlations</a>
            <a href="#quality">✅ Data Quality</a>
            <a href="#reports">📋 Detailed Reports</a>
        </div>"""
        
        # Dataset Overview Section
        html_content += f"""
        <div id="overview" class="section">
            <h2>📈 Dataset Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{df.shape[0]:,}</h3>
                    <p>Total Rows</p>
                </div>
                <div class="stat-card">
                    <h3>{df.shape[1]}</h3>
                    <p>Total Columns</p>
                </div>
                <div class="stat-card">
                    <h3>{df.isnull().sum().sum():,}</h3>
                    <p>Missing Values</p>
                </div>
                <div class="stat-card">
                    <h3>{len(df.select_dtypes(include=['number']).columns)}</h3>
                    <p>Numeric Columns</p>
                </div>
            </div>
            
            <h3>📋 Column Information</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Data Type</th>
                        <th>Non-Null Count</th>
                        <th>Missing %</th>
                        <th>Unique Values</th>
                    </tr>
                </thead>
                <tbody>"""
        
        # Add column information
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            html_content += f"""
                    <tr>
                        <td><strong>{col}</strong></td>
                        <td>{df[col].dtype}</td>
                        <td>{df[col].count():,}</td>
                        <td>{missing_pct:.1f}%</td>
                        <td>{df[col].nunique():,}</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </div>"""
        
        # Boxplot Analysis Section
        html_content += """
        <div id="boxplots" class="section">
            <h2>📦 Boxplot Analysis - Outlier Detection</h2>
            <p>Detailed boxplot analysis for each numerical variable showing outliers, quartiles, and statistical summaries.</p>
            
            <div class="boxplot-detail">
                <h3>🎯 Overview Boxplots</h3>
                <p>Comprehensive view of all numerical columns with outlier statistics.</p>
            </div>
            
            <div class="image-grid">
                <div class="image-container">
                    <img src="boxplot_overview.png" alt="Boxplot Overview">
                    <h3>Boxplot Overview</h3>
                    <p>All numerical columns with outlier counts and percentages</p>
                </div>
            </div>
            
            <div class="boxplot-detail">
                <h3>🔍 Individual Detailed Analysis</h3>
                <p>Each variable analyzed individually with boxplot + distribution + comprehensive statistics.</p>
            </div>
            
            <div class="image-grid">"""
        
        # Add individual boxplot images
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:6]:  # Limit to first 6 columns
            safe_col_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('$', 'dollar')
            
            # Calculate outlier statistics for the description
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            html_content += f"""
                <div class="image-container">
                    <img src="boxplot_detail_{safe_col_name}.png" alt="Detailed analysis of {col}">
                    <h3>{col}</h3>
                    <p><strong>{outlier_count}</strong> outliers ({outlier_pct:.1f}%) | IQR: {IQR:.2f}</p>
                    <p>Median: {df[col].median():.2f} | Mean: {df[col].mean():.2f}</p>
                </div>"""
        
        html_content += """
            </div>
        </div>"""
        
        # Violin Plot Analysis Section
        html_content += """
        <div id="violinplots" class="section">
            <h2>🎻 Violin Plot Analysis - Distribution Shape</h2>
            <p>Detailed violin plot analysis showing complete distribution shapes, density curves, and statistical measures including skewness and kurtosis.</p>
            
            <div class="boxplot-detail">
                <h3>🎵 Overview Violin Plots</h3>
                <p>Comprehensive view of distribution shapes for all numerical columns with skewness analysis.</p>
            </div>
            
            <div class="image-grid">
                <div class="image-container">
                    <img src="violin_overview.png" alt="Violin Plot Overview">
                    <h3>Violin Plot Overview</h3>
                    <p>All numerical columns with distribution shapes and skewness indicators</p>
                </div>
            </div>
            
            <div class="boxplot-detail">
                <h3>🔍 Individual Distribution Analysis</h3>
                <p>Each variable analyzed individually with violin plot + density curve + comprehensive distribution statistics.</p>
            </div>
            
            <div class="image-grid">"""
        
        # Add individual violin plot images
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:6]:  # Limit to first 6 columns
            safe_col_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('$', 'dollar')
            
            # Calculate distribution statistics for the description
            mean_val = df[col].mean()
            median_val = df[col].median()
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            
            # Skewness interpretation
            if abs(skewness) < 0.5:
                skew_desc = "Symmetric"
            elif skewness > 0.5:
                skew_desc = "Right-skewed"
            else:
                skew_desc = "Left-skewed"
            
            # Kurtosis interpretation
            if abs(kurtosis) < 0.5:
                kurt_desc = "Normal"
            elif kurtosis > 0.5:
                kurt_desc = "Heavy-tailed"
            else:
                kurt_desc = "Light-tailed"
            
            html_content += f"""
                <div class="image-container">
                    <img src="violin_detail_{safe_col_name}.png" alt="Detailed violin analysis of {col}">
                    <h3>{col}</h3>
                    <p><strong>{skew_desc}</strong> distribution | <strong>{kurt_desc}</strong> tails</p>
                    <p>Mean: {mean_val:.3f} | Median: {median_val:.3f} | Skew: {skewness:.3f}</p>
                </div>"""
        
        html_content += """
            </div>
        </div>"""
        
        # Distribution Analysis Section
        html_content += """
        <div id="distributions" class="section">
            <h2>📊 Distribution Analysis</h2>
            <p>Histogram distributions with KDE curves for all numerical variables.</p>
            
            <div class="image-grid">
                <div class="image-container">
                    <img src="distribution_overview.png" alt="Distribution Overview">
                    <h3>Distribution Overview</h3>
                    <p>Histograms with KDE curves showing data distribution patterns</p>
                </div>
            </div>
        </div>"""
        
        # Correlation Analysis Section
        html_content += """
        <div id="correlations" class="section">
            <h2>🔗 Correlation Analysis</h2>
            <p>Correlation matrix showing relationships between numerical variables.</p>
            
            <div class="image-grid">
                <div class="image-container">
                    <img src="corr_heatmap.png" alt="Correlation Heatmap">
                    <h3>Correlation Heatmap</h3>
                    <p>Pearson correlation coefficients between all numerical variables</p>
                </div>"""
        
        # Add feature importance if target column exists
        if target_col and target_col in df.columns:
            html_content += f"""
                <div class="image-container">
                    <img src="feature_importance.png" alt="Feature Importance">
                    <h3>Feature Importance</h3>
                    <p>Random Forest importance ranking for target: <strong>{target_col}</strong></p>
                </div>"""
        
        html_content += """
            </div>
        </div>"""
        
        # Data Quality Section
        html_content += """
        <div id="quality" class="section">
            <h2>✅ Data Quality Assessment</h2>
            <p>Comprehensive data quality analysis including missing values, data types, and uniqueness.</p>
            
            <div class="image-grid">
                <div class="image-container">
                    <img src="missing_matrix.png" alt="Missing Value Matrix">
                    <h3>Missing Value Matrix</h3>
                    <p>Visual pattern analysis of missing data across the dataset</p>
                </div>
                <div class="image-container">
                    <img src="data_quality_summary.png" alt="Data Quality Summary">
                    <h3>Data Quality Summary</h3>
                    <p>4-panel comprehensive quality assessment with statistics</p>
                </div>
            </div>
        </div>"""
        
        # Detailed Reports Section
        html_content += """
        <div id="reports" class="section">
            <h2>📋 Detailed Interactive Reports</h2>
            <p>Access comprehensive automated reports with advanced statistics and interactive features.</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>📈</h3>
                    <p><a href="pandas_profiling.html" target="_blank" style="color: white; text-decoration: none;">
                        <strong>Pandas Profiling Report</strong><br>
                        Complete statistical analysis
                    </a></p>
                </div>
                <div class="stat-card">
                    <h3>🍯</h3>
                    <p><a href="sweetviz_report.html" target="_blank" style="color: white; text-decoration: none;">
                        <strong>SweetViz Interactive Report</strong><br>
                        Interactive data exploration
                    </a></p>
                </div>
            </div>
        </div>
        
        <div class="section" style="text-align: center; background: #667eea; color: white;">
            <h2>🎉 Analysis Complete!</h2>
            <p>This enhanced EDA report provides comprehensive insights into your dataset.</p>
            <p><strong>Generated with ❤️ using Python EDA Acceleration Tool</strong></p>
            <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>
    </div>
    
    <script>
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
    </script>
</body>
</html>"""
        
        # Save HTML report
        output_file = os.path.join(output_path, "enhanced_eda_report.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Custom HTML report with embedded boxplots saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating custom HTML report: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive EDA reports from CSV/TXT files with automatic format detection"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input data file (CSV, TXT, TSV, DAT with any separator: comma, semicolon, tab, pipe, space)"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where HTML reports will be saved"
    )
    parser.add_argument(
        "--target",
        default="Sales",
        help="Target column name for feature importance (default: Sales)"
    )
    args = parser.parse_args()
    
    try:
        logger.info("Starting EDA report generation...")
        
        # Validate input file
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
            
        # Create output directory
        os.makedirs(args.outdir, exist_ok=True)
        logger.info(f"Output directory ready: {args.outdir}")

        # Load and validate dataset
        logger.info("Loading dataset...")
        logger.info(f"Input file: {args.input}")
        
        # Check file extension
        file_extension = os.path.splitext(args.input)[1].lower()
        supported_extensions = ['.csv', '.txt', '.tsv', '.dat']
        
        if file_extension not in supported_extensions:
            logger.warning(f"File extension '{file_extension}' not in typical list {supported_extensions}, but attempting to read anyway...")
        
        # Use smart CSV reader with automatic format detection
        df = smart_read_csv(args.input)
        
        if df is None:
            logger.error("Failed to read the input file. Please check the file format and try again.")
            sys.exit(1)
            
        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns detected: {', '.join(df.columns.tolist())}")
        
        # Data validation
        if df.empty:
            logger.error("Dataset is empty")
            sys.exit(1)
        
        # Check for suspicious column names (might indicate wrong separator)
        suspicious_patterns = [';', '|', '\t']
        for col in df.columns:
            if any(pattern in str(col) for pattern in suspicious_patterns):
                logger.warning(f"Column name '{col}' contains separator characters - double-check if file was read correctly")
        
        logger.info("✅ Data reading and validation completed successfully")
        
        # Summary stats
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            logger.info(f"Missing values detected: {missing_data} total")
        else:
            logger.info("No missing values found")

        # Generate visualizations
        logger.info("Generating visualizations...")
        vis_results = []
        vis_results.append(plot_missing_matrix(df, args.outdir))
        vis_results.append(plot_correlation_heatmap(df, args.outdir))
        vis_results.append(plot_distribution_overview(df, args.outdir))
        vis_results.append(plot_categorical_overview(df, args.outdir))
        vis_results.append(plot_boxplot_overview(df, args.outdir))
        vis_results.append(plot_individual_boxplots(df, args.outdir))
        vis_results.append(plot_violin_overview(df, args.outdir))
        vis_results.append(plot_individual_violin_plots(df, args.outdir))
        vis_results.append(plot_data_quality_summary(df, args.outdir))
        
        # Feature importance (check if target column exists)
        if args.target in df.columns:
            vis_results.append(plot_feature_importance(df, args.outdir, target_col=args.target))
        else:
            logger.warning(f"Target column '{args.target}' not found. Available columns: {df.columns.tolist()}")

        # Generate reports
        logger.info("Generating comprehensive reports...")
        report_results = []
        report_results.append(generate_pandas_profile(args.input, args.outdir))
        report_results.append(generate_sweetviz_report(args.input, args.outdir))
        
        # Generate custom HTML report with embedded boxplots
        logger.info("Creating enhanced HTML report with embedded visualizations...")
        report_results.append(generate_custom_html_report(df, args.outdir, args.target))
        
        # Final summary
        successful_vis = sum(vis_results)
        successful_reports = sum(report_results)
        
        logger.info(f"Visualization summary: {successful_vis}/{len(vis_results)} successful")
        logger.info(f"Report summary: {successful_reports}/{len(report_results)} successful")
        logger.info(f"EDA generation complete! Check directory: {args.outdir}")
        
        # List generated files
        if os.path.exists(args.outdir):
            files = os.listdir(args.outdir)
            if files:
                logger.info("Generated files:")
                for file in sorted(files):
                    file_path = os.path.join(args.outdir, file)
                    size = os.path.getsize(file_path) / 1024
                    logger.info(f"   - {file} ({size:.1f} KB)")
            else:
                logger.warning("No files were generated")
                
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
