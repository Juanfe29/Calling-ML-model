
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
from xgboost import XGBClassifier

OUTPUT_DIR = "analysis"
MODEL_DIR = "model"
DATA_FILE = "bank-additional-full.csv"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "advanced_modeling_results.md")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

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
    
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5, 6, 8],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'classifier__scale_pos_weight': [scale_pos_weight, scale_pos_weight * 0.5, scale_pos_weight * 1.5]
    }
    
    print("Starting Hyperparameter Tuning (RandomizedSearchCV)...")
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='f1',
        cv=cv_strategy,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best Params: {search.best_params_}")
    print(f"Best CV Score (F1): {search.best_score_:.4f}")
    
    best_model = search.best_estimator_
    
    model_path = os.path.join(MODEL_DIR, "best_xgboost.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    with open(RESULTS_FILE, "w") as f:
        f.write("# Advanced Modeling Results (Tuned XGBoost)\n\n")
        f.write("## Hyperparameter Tuning\n")
        f.write(f"- **Best Parameters:** `{search.best_params_}`\n")
        f.write(f"- **Best CV F1-Score:** {search.best_score_:.4f}\n\n")
        
        evaluate_model(best_model, X_test, y_test, "Tuned XGBoost", f)

    print(f"Advanced results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
