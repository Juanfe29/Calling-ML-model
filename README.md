# Term Deposit Subscription Predictor

This project uses a trained XGBoost model to predict whether a client will subscribe
to a term deposit product. The model was trained on the UCI Bank Marketing dataset
(41,188 records) and achieves a ROC-AUC of 0.81+ on the test set.


## Contents

    app.py                     Streamlit web application
    01_eda.py                  Exploratory data analysis script
    02_modeling.py             Baseline models (Logistic Regression, Random Forest, XGBoost)
    03_advanced_modeling.py    Tuned XGBoost with RandomizedSearchCV
    bank-additional-full.csv   Full dataset
    images/                    All charts and plots generated during analysis


## Requirements

Install dependencies before running anything:

    pip install streamlit pandas numpy scikit-learn xgboost joblib


## Running the Analysis Scripts

Run the scripts in order from the project root directory.
Each script saves its outputs (PNG plots, model file) automatically.

    python3 01_eda.py
    python3 02_modeling.py
    python3 03_advanced_modeling.py

After running 03_advanced_modeling.py, a trained model will be saved to model/best_xgboost.pkl.
The app requires this file to work.


## Running the App

From the project root directory:

    python3 -m streamlit run app.py

The app opens at http://localhost:8501


## App Usage

The app has two modes accessible via tabs at the top of the page.


### Manual Entry

Fill in the client fields across three columns: demographics, financial profile,
and campaign history. Click Predict to see the subscription probability,
predicted outcome, and priority tier (High, Medium, or Low).


### CSV Batch Prediction

Upload a CSV file with one row per client. The app accepts the same column format
as the original dataset. The duration column is ignored if present (data leakage).
The pdays column is automatically converted to a binary was_contacted flag.

After scoring, the results appear as a ranked table sorted by subscription probability.
Use the Download Ranked List button to export the scored file as a CSV ready for
the call center team.

Expected columns:

    age, job, marital, education, default, housing, loan, contact, month,
    day_of_week, campaign, pdays, previous, poutcome,
    emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed


## Notes

The model excludes the duration variable intentionally. Duration is only known
after the call ends, making it a data leakage source for any pre-call scoring task.
All categorical encoding and feature engineering are handled inside the model pipeline.
