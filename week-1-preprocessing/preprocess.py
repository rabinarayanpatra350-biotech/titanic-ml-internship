"""
Titanic Dataset - Cleaning & Preprocessing Pipeline
=====================================================
Week 1 Task: Python for Machine Learning & Data Preprocessing
(Internship project - Rabi Narayan Patra)

This script cleans and preprocesses the classic Titanic dataset,
documenting every preprocessing step:

  Step 1. Data loading and initial inspection
  Step 2. Exploratory data analysis (EDA)
  Step 3. Handling missing values
  Step 4. Data cleaning (duplicates, data types, outliers)
  Step 5. Feature engineering and feature selection
  Step 6. Encoding categorical variables
  Step 7. Normalization (min-max scaling)
  Step 8. Final verification and model-readiness check

Input : titanic.csv (891 passengers x 12 columns)
Output: cleaned_titanic.csv + figures/ (PNG charts)

Requirements: pandas, numpy, matplotlib, scikit-learn
Run: python preprocess.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)


# ----------------------------------------------------------------------
# STEP 1 - Data loading and initial inspection
# ----------------------------------------------------------------------
print("=" * 70)
print("STEP 1 - DATA LOADING AND INITIAL INSPECTION")
print("=" * 70)
df = pd.read_csv("titanic.csv")
print(f"Loaded titanic.csv -> shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nColumn overview (first 5 rows):")
print(df.head().to_string())

print("\nData types and non-null counts:")
print(df.info(memory_usage=False))

print("\nSummary statistics (numeric columns):")
print(df.describe().round(2).to_string())

print("\nMissing values per column:")
print(df.isnull().sum().to_string())

# Figure 1 - missing values bar chart
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(missing.index, missing.values, color="#2F6F8F")
for i, v in enumerate(missing.values):
    ax.text(i, v + 8, str(v), ha="center", fontsize=10)
ax.set_title("Missing Values per Column (before preprocessing)")
ax.set_ylabel("Count of missing values")
plt.tight_layout()
plt.savefig("figures/fig1_missing_values.png", dpi=110)
plt.close(fig)

# ----------------------------------------------------------------------
# STEP 2 - Exploratory data analysis (EDA)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print(f"Overall survival rate: {df['Survived'].mean():.1%}")
print("\nSurvival rate by Sex:")
print(df.groupby("Sex")["Survived"].agg(["count", "mean"]).round(3).to_string())
print("\nSurvival rate by Passenger Class:")
print(df.groupby("Pclass")["Survived"].agg(["count", "mean"]).round(3).to_string())
print("\nAge statistics:")
print(df["Age"].describe().round(2).to_string())

# Figure 2 - EDA panel (2 x 2)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
colors = ["#C0504D", "#4F81BD"]

surv_by_sex = df.groupby("Sex")["Survived"].mean()
axes[0, 0].bar(surv_by_sex.index, surv_by_sex.values, color=colors)
axes[0, 0].set_title("Survival rate by sex")
axes[0, 0].set_ylabel("Survival rate")
for i, v in enumerate(surv_by_sex.values):
    axes[0, 0].text(i, v + 0.02, f"{v:.0%}", ha="center")

surv_by_class = df.groupby("Pclass")["Survived"].mean()
axes[0, 1].bar(surv_by_class.index.astype(str), surv_by_class.values, color="#9BBB59")
axes[0, 1].set_title("Survival rate by passenger class")
axes[0, 1].set_xlabel("Pclass")
for i, v in enumerate(surv_by_class.values):
    axes[0, 1].text(i, v + 0.02, f"{v:.0%}", ha="center")

axes[1, 0].hist(df["Age"].dropna(), bins=30, color="#8064A2", edgecolor="white")
axes[1, 0].set_title("Age distribution")
axes[1, 0].set_xlabel("Age (years)")

axes[1, 1].hist(df["Fare"], bins=30, color="#4F81BD", edgecolor="white")
axes[1, 1].set_title("Fare distribution")
axes[1, 1].set_xlabel("Fare (pounds)")
axes[1, 1].set_yscale("log")

plt.suptitle("Exploratory data analysis - Titanic dataset", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig2_eda.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# STEP 3 - Handling missing values
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 - HANDLING MISSING VALUES")
print("=" * 70)

# 3a. 'Cabin' is missing for 687/891 passengers (77.1%) -> drop the column
cabin_pct = df["Cabin"].isnull().mean()
print(f"\n3a. Cabin: {df['Cabin'].isnull().sum()} missing ({cabin_pct:.1%})")
print("    -> Too much missing data to impute reliably; column dropped.")
df = df.drop(columns=["Cabin"])

# 3b. 'Age' (177 missing, 19.9%) -> impute with median age of the passenger's
#     (Sex, Pclass) group, because age differs systematically across groups.
print(f"\n3b. Age: {df['Age'].isnull().sum()} missing ({df['Age'].isnull().mean():.1%})")
group_medians = df.groupby(["Sex", "Pclass"])["Age"].median()
print("    Median age by (Sex, Pclass) group:")
print(group_medians.round(1).to_string())
age_before = df["Age"].copy()
df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
    lambda s: s.fillna(s.median()))
print(f"    -> Imputed with group medians. Remaining missing: {df['Age'].isnull().sum()}")

# Figure 3 - age distribution before vs after imputation
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].hist(age_before.dropna(), bins=30, color="#8064A2", edgecolor="white")
axes[0].set_title("Age - before imputation (n=714)")
axes[0].set_xlabel("Age (years)")
axes[1].hist(df["Age"], bins=30, color="#4F81BD", edgecolor="white")
axes[1].set_title("Age - after imputation (n=891)")
axes[1].set_xlabel("Age (years)")
axes[0].set_ylabel("Passengers")
plt.tight_layout()
plt.savefig("figures/fig3_imputation.png", dpi=110)
plt.close(fig)

# 3c. 'Embarked' (2 missing, 0.2%) -> impute with the mode (most common port)
mode_port = df["Embarked"].mode()[0]
print(f"\n3c. Embarked: 2 missing -> imputed with mode ('{mode_port}')")
df["Embarked"] = df["Embarked"].fillna(mode_port)

print(f"\nTotal missing values after Step 3: {df.isnull().sum().sum()}")

# ----------------------------------------------------------------------
# STEP 4 - Data cleaning: duplicates, data types, outliers
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 - DATA CLEANING (DUPLICATES, TYPES, OUTLIERS)")
print("=" * 70)

dup_mask = df.duplicated(subset=["Name"])
print(f"\nDuplicate rows found: {df.duplicated().sum()}")
print(f"Duplicate passenger names: {dup_mask.sum()} -> none removed (all unique)")
df = df.drop_duplicates()

df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")
print("\nData types after conversion: Sex and Embarked -> category")

q1, q3 = df["Fare"].quantile([0.25, 0.75])
iqr = q3 - q1
fare_outliers = ((df["Fare"] < q1 - 1.5 * iqr) | (df["Fare"] > q3 + 1.5 * iqr)).sum()
print(f"\nFare outliers (beyond 1.5 x IQR): {fare_outliers} ({fare_outliers/len(df):.1%})")
print("-> Kept: high fares are real first-class tickets, not errors.")
print("-> 'Fare' will be normalized in Step 7 instead of removed.")

# ----------------------------------------------------------------------
# STEP 5 - Feature engineering and feature selection
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 - FEATURE ENGINEERING AND SELECTION")
print("=" * 70)

# 5a. Extract honorific Title from Name, then simplify into 4 groups
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["Title"] = df["Title"].replace({
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare = df["Title"].value_counts()
rare_titles = rare[rare < 10].index
df["Title"] = df["Title"].where(~df["Title"].isin(rare_titles), "Rare")
print("\n5a. Extracted 'Title' from Name and grouped rare titles:")
print(df["Title"].value_counts().to_string())

# 5b. FamilySize = SibSp + Parch + 1 (the passenger); IsAlone flag
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
print(f"\n5b. FamilySize created; passengers travelling alone: "
      f"{df['IsAlone'].sum()} ({df['IsAlone'].mean():.0%})")

# 5c. Feature selection - drop identifiers / free-text columns not usable
#     by most machine-learning models
drop_cols = ["PassengerId", "Name", "Ticket"]
df = df.drop(columns=drop_cols)
print(f"\n5c. Dropped identifier/text columns: {drop_cols}")

# Figure 4 - survival by Title and FamilySize
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t = df.groupby("Title")["Survived"].mean().sort_values()
axes[0].barh(t.index, t.values, color="#2F6F8F")
axes[0].set_title("Survival rate by Title")
for i, v in enumerate(t.values):
    axes[0].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
fs = df.groupby("FamilySize")["Survived"].mean()
axes[1].bar(fs.index.astype(str), fs.values, color="#9BBB59")
axes[1].set_title("Survival rate by family size")
axes[1].set_xlabel("Family size")
plt.tight_layout()
plt.savefig("figures/fig4_features.png", dpi=110)
plt.close(fig)

# ----------------------------------------------------------------------
# STEP 6 - Encoding categorical variables
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6 - ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# 6a. Binary encoding for Sex (female=1, male=0)
df["Sex"] = df["Sex"].cat.codes          # female -> 0, male -> 1
print("\n6a. Sex encoded as binary integer (0 = female, 1 = male)")

# 6b. One-hot encoding for Embarked (3 ports) and Title (4 groups)
n_before = df.shape[1]
n_cat = sum(c.startswith(("Embarked", "Title")) for c in df.columns)
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=False)
n_dummies = df.shape[1] - (n_before - n_cat)
print(f"6b. One-hot encoding: {n_cat} categorical columns replaced by "
      f"{n_dummies} dummy (0/1) columns")
print("   New columns:", [c for c in df.columns if c.startswith(("Embarked", "Title"))])
print(f"\nAll columns are now numeric: {all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes)}")

# ----------------------------------------------------------------------
# STEP 7 - Normalization (min-max scaling)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7 - NORMALIZATION (MIN-MAX SCALING)")
print("=" * 70)

scale_cols = ["Age", "Fare"]
print("\nBefore scaling:")
print(df[scale_cols].describe().round(2).to_string())

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print("\nAfter min-max scaling (all values in [0, 1]):")
print(df[scale_cols].describe().round(3).to_string())

# Figure 5 - correlation heatmap of the final feature set
plt.figure(figsize=(9, 7))
corr = df.corr(numeric_only=True)
im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, label="Pearson correlation")
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(corr)), corr.columns, fontsize=8)
plt.title("Correlation heatmap - final preprocessed dataset")
plt.tight_layout()
plt.savefig("figures/fig5_correlation.png", dpi=110)
plt.close(fig)

# ----------------------------------------------------------------------
# STEP 8 - Final verification and model-readiness check
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8 - FINAL VERIFICATION AND MODEL-READINESS")
print("=" * 70)

print(f"\nFinal dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Non-numeric columns: {(~df.dtypes.apply(pd.api.types.is_numeric_dtype)).sum()}")
print("\nFinal columns:", list(df.columns))

df.to_csv("cleaned_titanic.csv", index=False)
print("\nSaved -> cleaned_titanic.csv")

# Model-readiness check: 5-fold cross-validated logistic regression
X = df.drop(columns=["Survived"])
y = df["Survived"]
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
scores = cross_val_score(model, X, y, cv=5)
print(f"\nModel-readiness check - 5-fold logistic regression accuracy: "
      f"{scores.mean():.1%} (+/- {scores.std():.1%})")
print("The preprocessed dataset loads into a scikit-learn pipeline without "
      "further modification.")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
