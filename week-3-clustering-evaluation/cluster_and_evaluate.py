"""
Titanic Dataset - Unsupervised Learning & Model Evaluation
===========================================================
Week 3 Task: clustering + dimensionality reduction + model evaluation
(Internship project - Rabi Narayan Patra)

PART A - Unsupervised learning (clustering)
  * K-Means with elbow method and silhouette analysis
  * Hierarchical (agglomerative, Ward) clustering with a dendrogram
  * PCA dimensionality reduction for visualization
  * Cluster profiling against the (unused) survival label

PART B - Model evaluation
  * Hyperparameter tuning of a Random Forest with GridSearchCV
  * Confusion matrix, precision, recall, F1, ROC-AUC on the test set
  * Tuned model vs the Week 2 baseline

Input : titanic.csv (same dataset as Weeks 1-2)
Output: cluster_profiles.csv, tuning_results.csv, figures/ (PNG)

Requirements: pandas, numpy, matplotlib, scikit-learn, scipy
Run: python cluster_and_evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)


# ======================================================================
# PART 0 - Load and prepare the data (Week 1 preprocessing, compact)
# ======================================================================
print("=" * 70)
print("PART 0 - DATA PREPARATION (reusing the Week 1 preprocessing)")
print("=" * 70)

df = pd.read_csv("titanic.csv")
df["FareRaw"] = df["Fare"].copy()          # raw fare, for cluster profiling

df = df.drop(columns=["Cabin"])
df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
    lambda s: s.fillna(s.median()))
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare = df["Title"].value_counts()
df["Title"] = df["Title"].where(~df["Title"].isin(rare[rare < 10].index), "Rare")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df = df.drop(columns=["PassengerId", "Name", "Ticket"])
df["Sex"] = df["Sex"].astype("category").cat.codes
df = pd.get_dummies(df, columns=["Embarked", "Title"])

print(f"Dataset after preparation: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")

# ======================================================================
# PART A1 - K-MEANS CLUSTERING (elbow + silhouette)
# ======================================================================
print("\n" + "=" * 70)
print("PART A1 - K-MEANS CLUSTERING")
print("=" * 70)

# Unsupervised: the survival label is removed before clustering.
# Features are standardized so no single scale dominates the distances.
X_cluster = df.drop(columns=["Survived", "FareRaw"]).copy()
X_scaled = StandardScaler().fit_transform(X_cluster)
print(f"Features for clustering: {X_cluster.shape[1]} (Survived excluded)")
print("Features standardized with StandardScaler before clustering.")

inertias, silhouettes = {}, {}
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    inertias[k] = km.inertia_
    silhouettes[k] = silhouette_score(X_scaled, km.labels_)

print("\n k | inertia   | silhouette")
for k in K_range:
    print(f"{k:2d} | {inertias[k]:9.0f} | {silhouettes[k]:.3f}")

best_k = max(silhouettes, key=silhouettes.get)
print(f"\nOptimal k by silhouette score: {best_k} "
      f"(silhouette = {silhouettes[best_k]:.3f})")

# Figure 1 - elbow and silhouette side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(K_range), [inertias[k] for k in K_range], "o-", color="#4F81BD")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[0].set_title("Elbow method")
axes[1].plot(list(K_range), [silhouettes[k] for k in K_range], "o-",
             color="#C0504D")
axes[1].axvline(best_k, ls="--", color="grey", alpha=0.6)
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title(f"Silhouette analysis (best k = {best_k})")
plt.tight_layout()
plt.savefig("figures/fig1_elbow_silhouette.png", dpi=110)
plt.close(fig)

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
df["Cluster"] = kmeans.labels_

# ======================================================================
# PART A2 - PCA DIMENSIONALITY REDUCTION + VISUALIZATION
# ======================================================================
print("\n" + "=" * 70)
print("PART A2 - PCA DIMENSIONALITY REDUCTION")
print("=" * 70)

pca = PCA(n_components=2).fit(X_scaled)
pcs = pca.transform(X_scaled)
evr = pca.explained_variance_ratio_
print(f"PC1 explains {evr[0]:.1%}, PC2 explains {evr[1]:.1%} "
      f"(together {evr.sum():.1%} of the variance)")

# Figure 2 - first two principal components, coloured by cluster
plt.figure(figsize=(8, 6))
colors = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#F79646"]
for c in range(best_k):
    m = df["Cluster"] == c
    plt.scatter(pcs[m, 0], pcs[m, 1], s=16, alpha=0.6, color=colors[c % 5],
                label=f"Cluster {c} (n={m.sum()})")
plt.xlabel(f"PC1 ({evr[0]:.0%} variance)")
plt.ylabel(f"PC2 ({evr[1]:.0%} variance)")
plt.title("K-Means clusters in PCA space (2 components)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig2_pca_clusters.png", dpi=110)
plt.close(fig)

# ======================================================================
# PART A3 - CLUSTER PROFILING (who is in each cluster?)
# ======================================================================
print("\n" + "=" * 70)
print("PART A3 - CLUSTER PROFILING")
print("=" * 70)

profile = df.groupby("Cluster").agg(
    passengers=("Survived", "size"),
    mean_age=("Age", "mean"),
    mean_fare=("FareRaw", "mean"),
    pct_female=("Sex", lambda s: (s == 0).mean()),
    pct_first_class=("Pclass", lambda s: (s == 1).mean()),
    survival_rate=("Survived", "mean"),
).round(2)
print(profile.to_string())
profile.to_csv("cluster_profiles.csv")
print("\nSaved -> cluster_profiles.csv")
print("(Survival was NOT shown to the clustering algorithm - it is used")
print("  here only to check what the clusters mean a posteriori.)")

# Figure 3 - cluster profile: survival rate and size
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(profile.index.astype(str), profile["survival_rate"],
            color=[colors[c % 5] for c in profile.index])
axes[0].axhline(df["Survived"].mean(), ls="--", color="grey", alpha=0.7,
                label=f"overall {df['Survived'].mean():.0%}")
axes[0].set_xlabel("K-Means cluster")
axes[0].set_ylabel("Survival rate")
axes[0].set_title("Survival rate by discovered cluster")
axes[0].legend(fontsize=9)
axes[1].bar(profile.index.astype(str), profile["mean_fare"],
            color=[colors[c % 5] for c in profile.index])
axes[1].set_xlabel("K-Means cluster")
axes[1].set_ylabel("Mean fare (\u00A3)")
axes[1].set_title("Mean fare by discovered cluster")
plt.tight_layout()
plt.savefig("figures/fig3_cluster_profile.png", dpi=110)
plt.close(fig)

# ======================================================================
# PART A4 - HIERARCHICAL CLUSTERING (Ward, dendrogram)
# ======================================================================
print("\n" + "=" * 70)
print("PART A4 - HIERARCHICAL CLUSTERING")
print("=" * 70)

Z = linkage(X_scaled, method="ward")
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X_scaled)
df["ClusterHier"] = agg.labels_
ari = adjusted_rand_score(df["Cluster"], df["ClusterHier"])
print(f"Agglomerative (Ward) clustering with k={best_k}")
print(f"Agreement with K-Means (adjusted Rand index): {ari:.3f}")

# Figure 4 - dendrogram (truncated to 40 leaves for readability)
plt.figure(figsize=(10, 4.5))
dendrogram(Z, truncate_mode="lastp", p=40, no_labels=True,
           color_threshold=Z[-best_k + 1, 2])
plt.title(f"Hierarchical clustering dendrogram (Ward linkage, cut at k={best_k})")
plt.ylabel("Merge distance")
plt.tight_layout()
plt.savefig("figures/fig4_dendrogram.png", dpi=110)
plt.close(fig)

# ======================================================================
# PART B - MODEL EVALUATION & HYPERPARAMETER TUNING
# ======================================================================
print("\n" + "=" * 70)
print("PART B - MODEL EVALUATION & HYPERPARAMETER TUNING")
print("=" * 70)

cls_df = df.drop(columns=["FareRaw", "Cluster", "ClusterHier"]).copy()
cls_df["Fare"] = MinMaxScaler().fit_transform(cls_df[["Fare"]])
X = cls_df.drop(columns=["Survived"])
y = cls_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train/test split: {len(X_train)} train / {len(X_test)} test (stratified)")

def evaluate(model, name):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    row = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    print(f"{name:28s} " + "  ".join(f"{v:.3f}" for v in row.values()))
    return row, pred, proba

# Baseline: the Week 2 random forest (default settings)
baseline = RandomForestClassifier(n_estimators=100,
                                  random_state=RANDOM_STATE).fit(X_train, y_train)
print("\nBaseline random forest (Week 2 settings), test set:")
base_row, _, _ = evaluate(baseline, "Baseline RF")

# Hyperparameter tuning with 5-fold grid search
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 5, 6, None],
    "min_samples_leaf": [1, 2],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE),
                    param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print(f"\nGrid search: {len(grid.cv_results_['params'])} parameter combinations"
      f" x {cv.n_splits} folds = {len(grid.cv_results_['params']) * cv.n_splits} fits")
print(f"Best parameters: {grid.best_params_}")
print(f"Best cross-validated F1 (train folds): {grid.best_score_:.3f}")

tuned_row, tuned_pred, tuned_proba = evaluate(grid.best_estimator_, "Tuned RF")

results = pd.DataFrame({"Baseline RF (Week 2)": base_row,
                        "Tuned RF (grid search)": tuned_row}).T.round(3)
print("\nModel evaluation comparison (test set):")
print(results.to_string())
results.to_csv("tuning_results.csv")
print("\nSaved -> tuning_results.csv")

# Figure 5 - confusion matrix + ROC curve of the tuned model
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
cm = confusion_matrix(y_test, tuned_pred)
im = axes[0].imshow(cm, cmap="Blues")
for r in range(2):
    for c in range(2):
        axes[0].text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=14,
                     color="white" if cm[r, c] > cm.max() / 2 else "black")
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("Tuned RF - confusion matrix (test set)")
fpr, tpr, _ = roc_curve(y_test, tuned_proba)
axes[1].plot(fpr, tpr, color="#2F6F8F",
             label=f"Tuned RF (AUC = {tuned_row['ROC-AUC']:.3f})")
axes[1].plot(fpr, tpr, color="#2F6F8F")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("Tuned RF - ROC curve (test set)")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig5_tuned_evaluation.png", dpi=110)
plt.close(fig)

print("\n" + "=" * 70)
print("CLUSTERING AND MODEL EVALUATION COMPLETE")
print("=" * 70)
