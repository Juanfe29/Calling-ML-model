import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

MODEL_PATH = "model/best_xgboost.pkl"

CATEGORICAL_COLS = {
    "job": ['admin.', 'blue-collar', 'technician', 'services', 'management',
            'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed',
            'student', 'unknown'],
    "marital": ['married', 'single', 'divorced', 'unknown'],
    "education": ['university.degree', 'high.school', 'basic.9y',
                  'professional.course', 'basic.4y', 'basic.6y', 'unknown', 'illiterate'],
    "default": ['no', 'unknown', 'yes'],
    "housing": ['yes', 'no', 'unknown'],
    "loan": ['no', 'yes', 'unknown'],
    "contact": ['cellular', 'telephone'],
    "month": ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    "day_of_week": ['mon', 'tue', 'wed', 'thu', 'fri'],
    "poutcome": ['nonexistent', 'failure', 'success'],
}

st.set_page_config(
    page_title="Term Deposit Prediction",
    page_icon="",
    layout="wide"
)

st.title(" Term Deposit  Subscription Predictor")
st.caption("Tuned XGBoost model | Trained on UCI UCI Marketing dataset (41,188 records)")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error("Model not found at `model/best_xgboost.pkl`. Run `03_advanced_modeling.py` first.")
    st.stop()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    if "pdays" in df.columns:
        df["was_contacted"] = (df["pdays"] < 999).astype(int)
        df = df.drop(columns=["pdays"])
    elif "was_contacted" not in df.columns:
        df["was_contacted"] = 0
    if "is_retired" not in df.columns:
        df["is_retired"] = (df["age"] > 60).astype(int)
    return df


def predict_df(df: pd.DataFrame):
    df_feat = engineer_features(df)
    proba = model.predict_proba(df_feat)[:, 1]
    pred = model.predict(df_feat)
    return proba, pred


tab_manual, tab_csv = st.tabs([" Manual Entry", " CSV Batch Prediction"])


with tab_manual:
    st.subheader("Single Client Prediction")

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox("Job", CATEGORICAL_COLS["job"])
            marital = st.selectbox("Marital Status", CATEGORICAL_COLS["marital"])
            education = st.selectbox("Education", CATEGORICAL_COLS["education"])

        with c2:
            st.markdown("**Financial Profile**")
            default = st.selectbox("Credit Default?", CATEGORICAL_COLS["default"])
            housing = st.selectbox("Housing Loan?", CATEGORICAL_COLS["housing"])
            loan = st.selectbox("Personal Loan?", CATEGORICAL_COLS["loan"])

            st.markdown("**Contact Info**")
            contact = st.selectbox("Contact Type", CATEGORICAL_COLS["contact"])
            month = st.selectbox("Last Contact Month", CATEGORICAL_COLS["month"], index=4)
            day_of_week = st.selectbox("Day of Week", CATEGORICAL_COLS["day_of_week"])

        with c3:
            st.markdown("**Campaign History**")
            campaign = st.number_input("Contacts This Campaign", 1, 50, 1)
            pdays_input = st.number_input("Days Since Last Contact (999 = never)", 0, 999, 999)
            previous = st.number_input("Contacts in Previous Campaigns", 0, 50, 0)
            poutcome = st.selectbox("Previous Campaign Outcome", CATEGORICAL_COLS["poutcome"])

            st.markdown("**Macro-Economic Indicators**")
            emp_var_rate = st.number_input("Employment Variation Rate", value=-1.8, step=0.1)
            cons_price_idx = st.number_input("Consumer Price Index", value=93.0, step=0.1)
            cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0, step=0.5)
            euribor3m = st.number_input("Euribor 3M Rate", value=1.0, step=0.01)
            nr_employed = st.number_input("Nr. Employees", value=5000.0, step=10.0)

        submitted = st.form_submit_button(" Predict", use_container_width=True)

    if submitted:
        was_contacted = 1 if pdays_input < 999 else 0
        is_retired = 1 if age > 60 else 0

        input_df = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day_of_week,
            "campaign": campaign, "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
            "nr.employed": nr_employed,
            "was_contacted": was_contacted, "is_retired": is_retired,
        }])

        try:
            proba, pred = predict_df(input_df)
            prob = proba[0]

            st.divider()
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Subscription Probability", f"{prob:.1%}")
            col_r2.metric("Predicted Outcome", " Subscribe" if pred[0] == 1 else " No Subscribe")
            col_r3.metric("Priority Tier",
                          " Top Priority" if prob >= 0.5 else
                          " Medium" if prob >= 0.25 else
                          " Low Priority")

            if pred[0] == 1:
                st.success("This client is likely to subscribe. Recommend adding to **Priority Call List**.")
            else:
                st.warning("Low subscription likelihood based on current profile.")

            with st.expander("Key signals used"):
                st.write({
                    "Previously contacted": "Yes" if was_contacted else "No",
                    "Is retired (age > 60)": "Yes" if is_retired else "No",
                    "Previous outcome": poutcome,
                    "Euribor 3M": euribor3m,
                    "Contacts this campaign": campaign,
                })

        except Exception as e:
            st.error(f"Prediction error: {e}")


with tab_csv:
    st.subheader("Batch Prediction from CSV")
    st.markdown(
        "Upload a CSV with client records. The app applies feature engineering automatically "
        "and returns a scored, ranked file ready for the call center."
    )

    with st.expander("Expected CSV columns"):
        st.code(
            "age, job, marital, education, default, housing, loan, contact, month,\n"
            "day_of_week, campaign, pdays, previous, poutcome,\n"
            "emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed\n\n"
            "Note: 'duration' is ignored if present. 'pdays' is converted to 'was_contacted'.",
            language="text"
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded, sep=None, engine="python")
            st.info(f"Loaded **{len(df_raw):,} records** with {df_raw.shape[1]} columns.")

            with st.spinner("Scoring all records..."):
                proba, pred = predict_df(df_raw)

            df_out = df_raw.copy()
            df_out["subscription_probability"] = np.round(proba, 4)
            df_out["predicted_subscribe"] = pred
            df_out["priority_tier"] = pd.cut(
                proba,
                bins=[-0.001, 0.25, 0.50, 1.001],
                labels=["Low", "Medium", "High"]
            )
            df_out = df_out.sort_values("subscription_probability", ascending=False).reset_index(drop=True)
            df_out.index += 1

            st.success(f"Done! {int(pred.sum())} / {len(pred)} clients predicted to subscribe.")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("High Priority (50%)", int((proba >= 0.5).sum()))
            col_m2.metric("Medium Priority (2550%)", int(((proba >= 0.25) & (proba < 0.5)).sum()))
            col_m3.metric("Low Priority (<25%)", int((proba < 0.25).sum()))

            st.dataframe(
                df_out[["subscription_probability", "predicted_subscribe", "priority_tier"]
                       + [c for c in df_out.columns if c not in
                          ["subscription_probability", "predicted_subscribe", "priority_tier"]]
                       ].head(200),
                use_container_width=True
            )

            csv_bytes = df_out.to_csv(index_label="rank").encode("utf-8")
            st.download_button(
                label=" Download Ranked List (CSV)",
                data=csv_bytes,
                file_name="ranked_call_list.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.divider()
st.caption("Model: Tuned XGBoost | Features: 20 input vars + was_contacted + is_retired | Duration excluded (data leakage)")
