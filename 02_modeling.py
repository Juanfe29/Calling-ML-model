
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score
from xgboost import XGBClassifier

OUTPUT_DIR = "analysis"
DATA_FILE = "bank-additional-full.csv"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "modeling_results.md")

def load_data(filepath):
    return pd.read_csv(filepath, sep=';')

def preprocess_data(df):
    if 'duration' in df.columns:
        print("Dropping 'duration' column (Leakage prevention)...")
        df = df.drop(columns=['duration'])
    
    print("Performing Feature Engineering...")
    
    if 'pdays' in df.columns:
        df['was_contacted'] = (df['pdays'] < 999).astype(int)
        df = df.drop(columns=['pdays'])
    
    if 'age' in df.columns:
        df['is_retired'] = (df['age'] > 60).astype(int)
    
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['y'])
    
    target = 'y'
    X = df.drop(columns=[target])
    y = df[target]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    return X, y, preprocessor

def evaluate_model(model, X_test, y_test, model_name, f):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"Evaluating {model_name}...")
    
    f.write(f"## {model_name} Results\n\n")
    
    report = classification_report(y_test, y_pred)
    f.write("### Classification Report\n")
    f.write(f"```\n{report}\n```\n\n")
    
    f1_class1 = f1_score(y_test, y_pred, pos_label=1)
    f.write(f"- **F1-Score (Class 1):** {f1_class1:.4f}\n\n")
    
    auc = roc_auc_score(y_test, y_prob)
    f.write(f"- **ROC AUC Score:** {auc:.4f}\n\n")
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"pr_curve_{model_name.replace(' ', '_')}.png"))
    plt.close()
    f.write(f"![Precision-Recall Curve {model_name}](pr_curve_{model_name.replace(' ', '_')}.png)\n\n")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{model_name.replace(' ', '_')}.png"))
    plt.close()
    f.write(f"![Confusion Matrix {model_name}](cm_{model_name.replace(' ', '_')}.png)\n\n")

    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            preprocessor = model.named_steps['preprocessor']
            try:
                cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
                num_names = preprocessor.named_transformers_['num'].feature_names_in_
                feature_names = np.concatenate([num_names, cat_names])
                
                importances = classifier.feature_importances_
                if len(importances) == len(feature_names):
                    indices = np.argsort(importances)[::-1][:20]
                    
                    plt.figure(figsize=(10, 8))
                    plt.title(f"Feature Importances - {model_name} (Top 20)")
                    plt.barh(range(len(indices)), importances[indices], align='center')
                    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                    plt.gca().invert_yaxis()
                    plt.xlabel('Relative Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, f"fi_{model_name.replace(' ', '_')}.png"))
                    plt.close()
                    f.write(f"![Feature Importance {model_name}](fi_{model_name.replace(' ', '_')}.png)\n\n")
            except Exception as e:
                print(f"Could not plot feature importance for {model_name}: {e}")

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = load_data(DATA_FILE)
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    with open(RESULTS_FILE, "w") as f:
        f.write("# Modeling Results\n\n")
        
        lr_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        
        lr_pipeline.fit(X_train, y_train)
        evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression", f)
        
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
        ])
        
        rf_pipeline.fit(X_train, y_train)
        evaluate_model(rf_pipeline, X_test, y_test, "Random Forest", f)
        
        xgb_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ))
        ])
        
        xgb_pipeline.fit(X_train, y_train)
        evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost", f)

    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
