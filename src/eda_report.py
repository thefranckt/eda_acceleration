import argparse
import os
import pandas as pd
from pandas_profiling import ProfileReport
import sweetviz as sv

def generate_pandas_profile(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    profile = ProfileReport(
        df,
        title="Pandas Profiling Report",
        explorative=True,
        correlations={"pearson": {"calculate": True}},
    )
    profile.to_file(os.path.join(output_path, "pandas_profiling.html"))

def generate_sweetviz_report(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    report = sv.analyze(df)
    report.show_html(
        filepath=os.path.join(output_path, "sweetviz_report.html"),
        open_browser=False
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate EDA reports with Pandas Profiling & SweetViz"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file (e.g. data/raw/data.csv)"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where HTML reports will be saved"
    )
    args = parser.parse_args()

    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(args.outdir, exist_ok=True)
    print(f"📂 Output directory ready: {args.outdir}")

    # Génère les rapports
    generate_pandas_profile(args.input, args.outdir)
    generate_sweetviz_report(args.input, args.outdir)
    
    print(f"✅ Reports generated in directory: {args.outdir}")

if __name__ == "__main__":
    main()
