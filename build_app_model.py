"""
Build a portable, deployment-friendly model artifact for the Streamlit app.

The notebook's FLAML AutoML selected an ExtraTrees model tuned for AUC.
We retrain a plain scikit-learn ExtraTreesClassifier with the FLAML-found
hyperparameters on the raw 13 features (no scaling), so the web app can take
plain patient inputs and only needs scikit-learn to load the model.
"""
import pickle
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURES = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
            'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']

df = pd.read_csv('heart.csv')
X = df[FEATURES]
y = df['output']

print("=== Feature stats (from heart.csv) ===")
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
for c in FEATURES:
    if c in CATEGORICAL:
        print(f"  {c:>8}: categorical unique = {sorted(df[c].unique().tolist())}")
    else:
        print(f"  {c:>8}: numeric min={df[c].min()} max={df[c].max()} mean={df[c].mean():.1f}")

# FLAML AutoML (AUC run) best hyperparameters, mapped to sklearn params:
#   n_estimators=37, max_features=0.2714, max_leaves->max_leaf_nodes=5, criterion=gini
params = dict(n_estimators=37, max_features=0.2714, max_leaf_nodes=5,
              criterion='gini', random_state=42)

# Honest holdout (same split as the notebook's AutoML section)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
m = ExtraTreesClassifier(**params).fit(Xtr, ytr)
proba = m.predict_proba(Xte)[:, 1]
print("\n=== Holdout (25%, random_state=42) ===")
print(f"  AUC = {roc_auc_score(yte, proba):.4f}")
print(f"  Accuracy = {accuracy_score(yte, m.predict(Xte)):.4f}")

# 5-fold CV (honest generalization estimate)
auc_cv = cross_val_score(ExtraTreesClassifier(**params), X, y, scoring='roc_auc', cv=5)
print("\n=== 5-fold CV ===")
print(f"  AUC = {auc_cv.mean():.4f} (+/- {auc_cv.std():.4f})")

# Final production model: train on ALL data, save portable pickle
final = ExtraTreesClassifier(**params).fit(X, y)
with open('app_model.pkl', 'wb') as f:
    pickle.dump(final, f)
print("\nSaved app_model.pkl (trained on all 303 rows, 13 features)")
print("Requires only scikit-learn to load:", type(final).__name__)
