"""
Titanic Survival Prediction API - Training & Serialization
============================================================
Week 4 Task (Capstone): end-to-end ML project
(Internship project - Rabi Narayan Patra)

Part 1 of the deployed system. This script:
  1. Loads titanic.csv and applies the Week 1 preprocessing
  2. Trains the Week 2 winner (logistic regression) and evaluates it
     on a held-out test set (accuracy, precision, recall, F1, ROC-AUC)
  3. Refits on the full dataset and serializes everything the API needs
     to model.pkl using joblib (model + fare scaler + feature list)

Input : titanic.csv
Output: model.pkl, model_metrics.csv, figures/ (2 charts)

Requirements: pandas, numpy, matplotlib, scikit-learn, joblib
Run: python train_and_serialize.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)

RANDOM_STATE = 42
import os
os.makedirs("figures", exist_ok=True)


# ----------------------------------------------------------------------
# Feature engineering - shared vocabulary between training and serving
# ----------------------------------------------------------------------
def engineer_features(df):
    """Apply the Week 1 preprocessing to a DataFrame that already has
    the raw Titanic columns. Returns (X, y, fare_scaler)."""
    df = df.drop(columns=["Cabin"], errors="ignore")
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda s: s.fillna(s.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    counts = df["Title"].value_counts()
    df["Title"] = df["Title"].where(~df["Title"].isin(counts[counts < 10].index), "Rare")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Sex"] = df["Sex"].astype("category").cat.codes
    df = pd.get_dummies(df, columns=["Embarked", "Title"])
    df = df.drop(columns=["PassengerId", "Name", "Ticket"], errors="ignore")

    scaler = MinMaxScaler().fit(df[["Fare"]].to_numpy())   # plain array: no feature-name warnings at serving time
    df["Fare"] = scaler.transform(df[["Fare"]].to_numpy())
    return df, scaler


# ======================================================================
# 1. Load and prepare
# ======================================================================
print("=" * 70)
print("STEP 1 - LOAD AND PREPARE THE DATA")
print("=" * 70)
df = pd.read_csv("titanic.csv")
features, fare_scaler = engineer_features(df)
X = features.drop(columns=["Survived"])
y = features["Survived"]
print(f"Prepared dataset: {X.shape[0]} rows x {X.shape[1]} features")

# ======================================================================
# 2. Train and evaluate (held-out test set)
# ======================================================================
print("\n" + "=" * 70)
print("STEP 2 - TRAIN AND EVALUATE THE MODEL")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "ROC-AUC": roc_auc_score(y_test, proba),
}
print("Test-set performance of the deployed model:")
for k, v in metrics.items():
    print(f"  {k:10s} {v:.3f}")
pd.DataFrame([metrics]).round(3).to_csv("model_metrics.csv", index=False)

# Figure - confusion matrix + ROC curve
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
cm = confusion_matrix(y_test, pred)
axes[0].imshow(cm, cmap="Blues")
for r in range(2):
    for c in range(2):
        axes[0].text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=14,
                      color="white" if cm[r, c] > cm.max() / 2 else "black")
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion matrix (accuracy {metrics['Accuracy']:.1%})")
fpr, tpr, _ = roc_curve(y_test, proba)
axes[1].plot(fpr, tpr, color="#2F6F8F",
             label=f"ROC curve (AUC = {metrics['ROC-AUC']:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve - deployed model")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig_model_performance.png", dpi=110)
plt.close(fig)

# Figure - logistic regression coefficients (model interpretation)
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
top = pd.concat([coefs.head(5), coefs.tail(5)])
plt.figure(figsize=(8, 5))
colors = ["#C0504D" if v < 0 else "#4F81BD" for v in top.values]
plt.barh(top.index, top.values, color=colors)
plt.axvline(0, color="black", lw=0.8)
plt.xlabel("Logistic regression coefficient")
plt.title("What the deployed model has learned\n(negative = lowers survival odds, positive = raises)")
plt.tight_layout()
plt.savefig("figures/fig_coefficients.png", dpi=110)
plt.close(fig)
print("\nStrongest model coefficients:")
print(top.round(3).to_string())

# ======================================================================
# 3. Refit on all data and serialize with joblib
# ======================================================================
print("\n" + "=" * 70)
print("STEP 3 - REFIT AND SERIALIZE (joblib)")
print("=" * 70)
final_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
final_model.fit(X, y)
bundle = {
    "model": final_model,
    "fare_scaler": fare_scaler,
    "features": list(X.columns),
    "metrics": {k: round(v, 3) for k, v in metrics.items()},
    "trained_on": "titanic.csv (891 passengers)",
    "library": f"scikit-learn LogisticRegression",
}
joblib.dump(bundle, "model.pkl")
print("Serialized -> model.pkl")
print(f"Bundle contents: model, fare_scaler, feature list ({len(bundle['features'])} features), metrics")
size_kb = os.path.getsize("model.pkl") / 1024
print(f"Serialized model size: {size_kb:.1f} KB")

# Round-trip verification: load the pickle and confirm identical predictions
loaded = joblib.load("model.pkl")
check = np.array_equal(loaded["model"].predict(X[:20]), final_model.predict(X[:20]))
print(f"Round-trip verification (load pickle, compare predictions): {'PASS' if check else 'FAIL'}")

print("\n" + "=" * 70)
print("MODEL READY FOR DEPLOYMENT - run 'python app.py' to serve it")
print("=" * 70)
