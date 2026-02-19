
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

OUTPUT_DIR = "analysis"
REPORT_FILE = os.path.join(OUTPUT_DIR, "eda_report.md")
DATA_FILE = "bank-additional-full.csv"

def load_data(filepath):
    return pd.read_csv(filepath, sep=';')

def generate_report(df):
    with open(REPORT_FILE, "w") as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        
        f.write("## 1. Data Structure\n")
        f.write(f"- **Rows:** {df.shape[0]}\n")
        f.write(f"- **Columns:** {df.shape[1]}\n")
        f.write("\n### Columns & Data Types\n")
        f.write(df.dtypes.to_markdown())
        f.write("\n\n")

        f.write("## 2. Missing Values\n")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            f.write("No missing values detected in the dataset.\n")
        else:
            f.write(missing[missing > 0].to_markdown())
        f.write("\n\n")
        
        f.write("### 'Unknown' Values Count\n")
        unknown_counts = (df == 'unknown').sum()
        f.write(unknown_counts[unknown_counts > 0].to_markdown())
        f.write("\n\n")

        f.write("## 3. Target Variable Distribution (`y`)\n")
        target_counts = df['y'].value_counts(normalize=True)
        f.write(target_counts.to_markdown())
        f.write("\n\n")
        
        distribucion = df['y'].value_counts(normalize=True) * 100
        f.write(f"Porcentaje de 'no': {distribucion['no']:.2f}%\n")
        f.write(f"Porcentaje de 'yes': {distribucion['yes']:.2f}%\n\n")
        
        plt.figure()
        sns.countplot(x='y', data=df)
        plt.title("Target Variable Distribution")
        plt.savefig(os.path.join(OUTPUT_DIR, "target_dist.png"))
        plt.close()
        f.write("![Target Distribution](target_dist.png)\n\n")

        f.write("## 4. Numerical Features Analysis\n")
        num_cols = df.select_dtypes(include=[np.number]).columns
        f.write(df[num_cols].describe().to_markdown())
        f.write("\n\n")

        plt.figure(figsize=(10, 8))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix (Numerical Features)")
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
        plt.close()
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

        f.write("## 5. Categorical Features Analysis\n")
        cat_cols = df.select_dtypes(include=['object']).columns
        cat_cols = [c for c in cat_cols if c != 'y']

        for col in cat_cols:
            f.write(f"### {col}\n")
            f.write(df[col].value_counts().to_markdown())
            f.write("\n\n")
            
            plt.figure(figsize=(10, 5))
            order = df[col].value_counts().index
            sns.countplot(y=col, data=df, order=order)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{col}.png"))
            plt.close()
            f.write(f"![{col} Distribution](dist_{col}.png)\n\n")

            plt.figure(figsize=(10, 5))
            sns.countplot(y=col, hue='y', data=df, order=order)
            plt.title(f"{col} vs Target")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"target_vs_{col}.png"))
            plt.close()
            f.write(f"![{col} vs Target](target_vs_{col}.png)\n\n")
        
        f.write("## 6. Business Insights - Numeric Variables vs Target\n")
        for col in num_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='y', y=col, data=df)
            plt.title(f"{col} by Target (y)")
            plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{col}.png"))
            plt.close()
            f.write(f"![{col} Boxplot](boxplot_{col}.png)\n\n")

    print(f"Report generated at {REPORT_FILE}")

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = load_data(DATA_FILE)
    generate_report(df)

if __name__ == "__main__":
    main()
