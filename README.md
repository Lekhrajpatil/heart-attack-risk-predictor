# ❤️ Heart Attack Risk Predictor

A Streamlit web app that estimates a patient's probability of a **higher chance of heart attack** from 13 clinical features, using an AutoML-tuned Extra Trees classifier trained on the UCI Heart dataset (303 patients).

> ⚠️ **Disclaimer:** This project is for educational/demo purposes only. It is **not** a medical device and provides **no medical advice**. Always consult a qualified healthcare professional.

## Live model performance

| Metric | Value |
|---|---|
| Model | `ExtraTreesClassifier` (tuned via AutoML) |
| Features | 13 (raw, unscaled — trees need no scaling) |
| Holdout AUC (25%) | 0.90 |
| 5-fold CV AUC | 0.91 ± 0.02 |

The model is saved as `app_model.pkl` and loads with **scikit-learn only** (no heavy AutoML dependency at runtime). See `build_app_model.py` for how it was trained.

## Dataset features (`heart.csv`)

| Feature | Type | Range / Values |
|---|---|---|
| `age` | numeric | 29–77 |
| `sex` | categorical | 0=Female, 1=Male |
| `cp` | categorical | 0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic |
| `trtbps` | numeric | resting blood pressure (mm Hg) |
| `chol` | numeric | cholesterol (mg/dl) |
| `fbs` | categorical | fasting blood sugar > 120 mg/dl (0/1) |
| `restecg` | categorical | 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy |
| `thalachh` | numeric | max heart rate achieved (bpm) |
| `exng` | categorical | exercise-induced angina (0/1) |
| `oldpeak` | numeric | ST depression (0.0–6.2) |
| `slp` | categorical | 0=Upsloping, 1=Flat, 2=Downsloping |
| `caa` | categorical | number of major vessels (0–4) |
| `thall` | categorical | 0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Other |
| `output` | target | 1 = more chance of heart attack, 0 = less chance |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

The app has two modes:
- **Single patient** — a form to enter 13 features and get a risk probability + category (Low / Moderate / High).
- **Batch (upload CSV)** — upload a CSV with the 13 feature columns to score many patients and download predictions.

## Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub (public).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app** → choose this repository and branch.
4. **Main file path:** `app.py`
5. **Python version:** 3.13 (3.12 also works). Dependencies install automatically from `requirements.txt`.
6. Click **Deploy**. Streamlit gives you a public URL like `https://<app-name>.streamlit.app`.

## Project structure

```
├── app.py                          # Streamlit web app
├── app_model.pkl                    # trained ExtraTrees model (scikit-learn only)
├── heart.csv                        # UCI Heart dataset
├── build_app_model.py              # script that trains & saves app_model.pkl
├── requirements.txt
├── README.md
├── Heart_Attack_Risk_Predictor_local.ipynb            # locally-runnable notebook (clean)
├── Heart_Attack_Risk_Predictor_local_executed.ipynb   # notebook with all outputs embedded
└── Heart_Attack_Risk_Predictor with Eval ML.ipynb     # original (Google Colab) notebook
```

## How this was built

The original notebook used **EvalML** for AutoML, which is incompatible with modern Python (requires Python ≤ 3.10 and older pandas/numpy/scikit-learn). The AutoML section was reproduced with **FLAML** (runs on Python 3.13), whose best Extra Trees configuration (`n_estimators=37, max_features≈0.27, max_leaf_nodes=5, criterion=gini`) was then retrained as a portable scikit-learn model for deployment.
