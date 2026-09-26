"""
Titanic Dataset - Supervised Machine Learning Models
=====================================================
Week 2 Task: Regression and Classification with scikit-learn
(Internship project - Rabi Narayan Patra)

Building on the Week 1 preprocessing pipeline, this script trains and
compares supervised learning models:

  CLASSIFICATION - predict Survived (0/1)
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * K-Nearest Neighbors
    Metrics: accuracy, precision, recall, F1, ROC-AUC + confusion matrices

  REGRESSION - predict Fare (pounds)
    * Linear Regression
    * Random Forest Regressor
    * K-Nearest Neighbors Regressor
    Metrics: MAE, RMSE, R-squared

Input : titanic.csv (same dataset as Week 1)
Output: classification_results.csv, regression_results.csv, figures/ (PNG)

Requirements: pandas, numpy, matplotlib, scikit-learn
Run: python train_models.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)

RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)


# ======================================================================
# PART 0 - Load and prepare the data (Week 1 preprocessing, compact)
# ======================================================================
print("=" * 70)
print("PART 0 - DATA PREPARATION (reusing the Week 1 preprocessing)")
print("=" * 70)

df = pd.read_csv("titanic.csv")
df["FareRaw"] = df["Fare"].copy()          # keep raw fare for the regression target

# Missing values (same decisions as Week 1)
df = df.drop(columns=["Cabin"])            # 77.1% missing
df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
    lambda s: s.fillna(s.median()))        # group medians
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Feature engineering (same as Week 1)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare = df["Title"].value_counts()
df["Title"] = df["Title"].where(~df["Title"].isin(rare[rare < 10].index), "Rare")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df = df.drop(columns=["PassengerId", "Name", "Ticket"])

# Encoding
df["Sex"] = df["Sex"].astype("category").cat.codes
df = pd.get_dummies(df, columns=["Embarked", "Title"])

print(f"Dataset after preparation: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")

# ----------------------------------------------------------------------
# CLASSIFICATION DATASET - the Week 1 cleaned dataset (Fare normalized)
# ----------------------------------------------------------------------
cls_df = df.drop(columns=["FareRaw"]).copy()
cls_df["Fare"] = MinMaxScaler().fit_transform(cls_df[["Fare"]])
X_cls = cls_df.drop(columns=["Survived"])
y_cls = cls_df["Survived"]

# ----------------------------------------------------------------------
# REGRESSION DATASET - target is the raw fare, log-transformed for
# training (fare is heavily right-skewed: median \u00A314.45, max \u00A3512.33).
# Predictions are transformed back to pounds before computing metrics.
# ----------------------------------------------------------------------
reg_df = df.drop(columns=["Fare"]).rename(columns={"FareRaw": "Fare"}).copy()
X_reg = reg_df.drop(columns=["Fare"])
y_reg = reg_df["Fare"]
y_reg_log = np.log1p(y_reg)

print(f"Classification features: {X_cls.shape[1]} | Regression features: {X_reg.shape[1]}")

# ======================================================================
# PART 1 - CLASSIFICATION: predict Survived
# ======================================================================
print("\n" + "=" * 70)
print("PART 1 - CLASSIFICATION (target: Survived)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=RANDOM_STATE)
print(f"Train/test split: {len(X_train)} train / {len(X_test)} test (stratified)")
print(f"Class balance (train): {y_train.mean():.1%} survived")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cls_results, fitted, cms, rocs = {}, {}, {}, {}

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv).mean()
    cls_results[name] = {
        "CV accuracy (train)": cv_acc,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    fitted[name] = model
    cms[name] = confusion_matrix(y_test, pred)
    rocs[name] = roc_curve(y_test, proba)

cls_table = pd.DataFrame(cls_results).T.round(3)
print("\nClassification results (test set):")
print(cls_table.to_string())
best_cls = max(cls_results, key=lambda k: cls_results[k]["F1"])
print(f"\nBest classifier by F1: {best_cls}")
cls_table.to_csv("classification_results.csv")

# Figure 1 - grouped bar chart of test metrics
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metrics))
w = 0.2
colors = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2"]
for i, (name, res) in enumerate(cls_results.items()):
    ax.bar(x + (i - 1.5) * w, [res[m] for m in metrics], w, label=name, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Classification metrics on the held-out test set")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig1_cls_metrics.png", dpi=110)
plt.close(fig)

# Figure 2 - confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, (name, cm) in zip(axes.flat, cms.items()):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name, fontsize=10)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=13,
                    color="white" if cm[r, c] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion matrices - test set (0 = died, 1 = survived)", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig2_confusion.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# Figure 3 - ROC curves
plt.figure(figsize=(7, 5))
for (name, (fpr, tpr, _)), color in zip(rocs.items(), colors):
    auc = cls_results[name]["ROC-AUC"]
    plt.plot(fpr, tpr, color=color, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves - test set")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3_roc.png", dpi=110)
plt.close(fig)

# ======================================================================
# PART 2 - REGRESSION: predict Fare
# ======================================================================
print("\n" + "=" * 70)
print("PART 2 - REGRESSION (target: Fare, pounds)")
print("=" * 70)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg_log, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train/test split: {len(Xr_train)} train / {len(Xr_test)} test")
yr_train_pounds = np.expm1(yr_train)          # for reporting
yr_test_pounds = np.expm1(yr_test)              # actual fares in pounds
print(f"Fare (target): mean \u00A3{y_reg.mean():.2f}, median \u00A3{y_reg.median():.2f}, max \u00A3{y_reg.max():.2f}")
print("Note: fare is heavily right-skewed, so models train on log1p(fare);")
print("      predictions are converted back to pounds before evaluation.")

regressors = {
    # KNN needs comparable feature scales -> scaler in a pipeline
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                             random_state=RANDOM_STATE),
    "K-Nearest Neighbors": Pipeline([("scale", StandardScaler()),
                                     ("knn", KNeighborsRegressor(n_neighbors=5))]),
}

reg_results, reg_preds = {}, {}

for name, model in regressors.items():
    model.fit(Xr_train, yr_train)
    pred = np.expm1(model.predict(Xr_test))    # back to pounds
    reg_results[name] = {
        "MAE": mean_absolute_error(yr_test_pounds, pred),
        "RMSE": np.sqrt(mean_squared_error(yr_test_pounds, pred)),
        "R2": r2_score(yr_test_pounds, pred),
    }
    reg_preds[name] = pred

reg_table = pd.DataFrame(reg_results).T.round(2)
print("\nRegression results (test set, pounds):")
print(reg_table.to_string())
best_reg = max(reg_results, key=lambda k: reg_results[k]["R2"])
print(f"\nBest regressor by R-squared: {best_reg}")
reg_table.to_csv("regression_results.csv")

# Figure 4 - predicted vs actual fare
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, (name, pred) in zip(axes, reg_preds.items()):
    ax.scatter(yr_test_pounds, pred, s=14, alpha=0.6, color=colors[list(reg_preds).index(name)])
    lims = [0, 300]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_title(f"{name}\nR\u00B2 = {reg_results[name]['R2']:.2f}, "
                f"MAE = \u00A3{reg_results[name]['MAE']:.1f}", fontsize=9)
    ax.set_xlabel("Actual fare (\u00A3)")
axes[0].set_ylabel("Predicted fare (\u00A3)")
plt.suptitle("Regression on fare - predicted vs actual (dashed line = perfect)", y=1.03)
plt.tight_layout()
plt.savefig("figures/fig4_reg_scatter.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# PART 3 - INTERPRETING THE BEST MODELS
# ======================================================================
print("\n" + "=" * 70)
print("PART 3 - MODEL INTERPRETATION")
print("=" * 70)

importances = pd.Series(fitted["Random Forest"].feature_importances_,
                        index=X_cls.columns).sort_values()
top = importances.tail(8)
print("Top feature importances (Random Forest classifier):")
print(importances.tail(8).sort_values(ascending=False).round(3).to_string())

plt.figure(figsize=(8, 5))
plt.barh(top.index, top.values, color="#2F6F8F")
plt.xlabel("Importance")
plt.title("What drives survival predictions? (Random Forest)")
plt.tight_layout()
plt.savefig("figures/fig5_feature_importance.png", dpi=110)
plt.close(fig)

dt = fitted["Decision Tree"]
print(f"\nDecision Tree depth: {dt.get_depth()} (max_depth=5 to limit overfitting)")
rf_train = accuracy_score(y_train, fitted["Random Forest"].predict(X_train))
rf_test = accuracy_score(y_test, fitted["Random Forest"].predict(X_test))
print(f"Random Forest train vs test accuracy: {rf_train:.3f} vs {rf_test:.3f}")

print("\n" + "=" * 70)
print("ALL MODELS TRAINED AND COMPARED - COMPLETE")
print("=" * 70)
