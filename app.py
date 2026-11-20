"""
Heart Attack Risk Predictor - Streamlit web app

Uses an ExtraTrees classifier (tuned via AutoML on the UCI Heart dataset) to
estimate the probability that a patient has a higher chance of heart attack
(dataset target: 1 = more chance, 0 = less chance).

Run locally:
    streamlit run app.py
"""
import pickle
import io

import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
            'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']

# Human-readable labels for categorical features (value -> label)
LABELS = {
    'sex': {0: 'Female', 1: 'Male'},
    'cp': {0: 'Typical angina', 1: 'Atypical angina', 2: 'Non-anginal pain', 3: 'Asymptomatic'},
    'fbs': {0: '<= 120 mg/dl', 1: '> 120 mg/dl'},
    'restecg': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'},
    'exng': {0: 'No', 1: 'Yes'},
    'slp': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'},
    'caa': {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels', 4: '4 vessels'},
    'thall': {0: 'Normal', 1: 'Fixed defect', 2: 'Reversible defect', 3: 'Other'},
}

# Feature order grouped for the form
NUMERIC = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

HELP = {
    'age': 'Age of the patient (years)',
    'sex': 'Sex of the patient',
    'cp': 'Chest pain type',
    'trtbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl',
    'restecg': 'Resting electrocardiographic results',
    'thalachh': 'Maximum heart rate achieved (bpm)',
    'exng': 'Exercise-induced angina',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slp': 'Slope of the peak exercise ST segment',
    'caa': 'Number of major vessels (0-4) coloured by fluoroscopy',
    'thall': 'Thalassemia status',
}


@st.cache_resource
def load_assets():
    model = pickle.load(open('app_model.pkl', 'rb'))
    df = pd.read_csv('heart.csv')
    return model, df


# ----- page config (must be the first Streamlit command in the script) -----
st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️",
                   layout="centered", initial_sidebar_state="expanded")

model, df = load_assets()

st.title("❤️ Heart Attack Risk Predictor")
st.caption("Estimates a patient's probability of a higher chance of heart attack "
           "from 13 clinical features, using an AutoML-tuned Extra Trees model "
           "(5-fold CV AUC ≈ 0.91).")

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Model:** ExtraTreesClassifier (AutoML-tuned)\n"
        "- **Data:** UCI Heart dataset (303 patients)\n"
        "- **Target:** 1 = more chance of heart attack, 0 = less chance\n"
        "- **CV AUC:** ~0.91  |  **Holdout AUC:** ~0.90\n\n"
        "This tool is for educational/demo purposes only and is **not** medical advice."
    )
    st.divider()
    st.markdown("Predicted value is the model's probability of **class 1** "
                "(higher chance of heart attack).")

tab_single, tab_batch = st.tabs(["🩺 Single patient", "📄 Batch (upload CSV)"])


def risk_category(p):
    if p < 0.40:
        return "Low", "🟢"
    if p < 0.70:
        return "Moderate", "🟠"
    return "High", "🔴"


def predict(df_in: pd.DataFrame) -> np.ndarray:
    X = df_in.reindex(columns=FEATURES)
    return model.predict_proba(X)[:, 1]


# ----------------- Single patient -----------------
with tab_single:
    with st.form("patient_form"):
        st.subheader("Enter patient details")
        cols = st.columns(3)
        inputs = {}

        # numeric inputs
        num_defaults = {c: float(df[c].median()) for c in NUMERIC}
        for i, c in enumerate(NUMERIC):
            with cols[i % 3]:
                if c == 'oldpeak':
                    inputs[c] = st.slider(f"{c}", min_value=float(df[c].min()),
                                          max_value=float(df[c].max()),
                                          value=num_defaults[c], step=0.1, help=HELP[c])
                else:
                    inputs[c] = st.slider(f"{c}", min_value=int(df[c].min()),
                                           max_value=int(df[c].max()),
                                           value=int(num_defaults[c]), step=1, help=HELP[c])

        st.divider()
        # categorical inputs
        cols2 = st.columns(4)
        for i, c in enumerate(CATEGORICAL):
            options = LABELS[c]
            keys = sorted(options.keys())
            labels = [options[k] for k in keys]
            default_label = options[int(df[c].mode().iloc[0])]
            with cols2[i % 4]:
                label = st.selectbox(f"{c}", options=labels, index=labels.index(default_label),
                                     help=HELP[c])
                inputs[c] = keys[labels.index(label)]

        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        row = pd.DataFrame([inputs])
        prob = float(predict(row)[0])
        cat, icon = risk_category(prob)
        st.divider()
        c1, c2 = st.columns([2, 3])
        c1.metric("Predicted risk", f"{prob*100:.1f}%", delta=cat, delta_color="inverse")
        c2.markdown(f"### {icon} {cat} risk")
        st.progress(prob, text=f"Probability of higher heart-attack chance: {prob*100:.1f}%")
        st.caption("Class 1 = more chance of heart attack; Class 0 = less chance.")
        with st.expander("Input features sent to the model"):
            st.dataframe(row[FEATURES].T.rename(columns={0: "value"}), use_container_width=True)

# ----------------- Batch prediction -----------------
with tab_batch:
    st.subheader("Upload a CSV with the 13 feature columns")
    st.markdown("Required columns: `" + "`, `".join(FEATURES) + "`  "
                "(an extra `output` column, if present, is ignored).")
    sample_cols = ", ".join(FEATURES)
    st.download_button("Download example CSV template",
                       data=df[FEATURES].head(5).to_csv(index=False),
                       file_name="heart_input_template.csv", mime="text/csv")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            batch = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                probs = predict(batch)
                out = batch.copy()
                out['risk_probability'] = probs
                out['risk_category'] = [risk_category(p)[0] for p in probs]
                n_high = int((probs >= 0.70).sum())
                n_mod = int(((probs >= 0.40) & (probs < 0.70)).sum())
                n_low = int((probs < 0.40).sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Patients", len(out))
                m2.metric("High risk", n_high)
                m3.metric("Low risk", n_low)
                st.dataframe(out, use_container_width=True)
                st.download_button("Download predictions",
                                   data=out.to_csv(index=False),
                                   file_name="heart_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
