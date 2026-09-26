# Titanic Machine Learning Internship — Complete Portfolio

**Author: Rabi Narayan Patra**

The complete, four-week AI/ML internship project in one repository: the classic Titanic
survival dataset taken through the **full machine-learning lifecycle** — from raw,
missing-riddled data to a deployed prediction API. Each week lives in its own folder
with the exact code that was run, its real run log, and its requirements.

| Week | Folder | What was done | Headline result |
|------|--------|---------------|-----------------|
| 1 — Preprocessing | [`week-1-preprocessing/`](week-1-preprocessing/) | 8 documented cleaning steps: missing values, EDA, feature engineering (Title, FamilySize, IsAlone), encoding, min–max scaling | 891×12 → 891×17 clean dataset, 0 missing, 82.2% CV accuracy |
| 2 — Supervised models | [`week-2-supervised-models/`](week-2-supervised-models/) | 4 classifiers + 3 regressors trained and compared | Logistic regression wins: **84.9% accuracy, 0.797 F1, 0.876 ROC-AUC** |
| 3 — Unsupervised & evaluation | [`week-3-clustering-evaluation/`](week-3-clustering-evaluation/) | K-Means (elbow + silhouette), Ward hierarchical clustering, PCA, cluster profiling, GridSearchCV tuning | Best k=8 (silhouette 0.417); clusters rediscover "women and children first"; tuned RF 82.1% |
| 4 — Deployment (capstone) | [`week-4-deployment-api/`](week-4-deployment-api/) | Model serialization with **joblib** + **Flask prediction API** with health check, validation, and automated tests | Live API: 2.2 KB model.pkl served over HTTP; all endpoint tests pass |

## How the weeks connect

Week 1 produces the cleaned feature set → Week 2 picks the best model on it → Week 3
validates the data's structure and tunes a competitor → Week 4 serializes the winner
and serves it. The deployed API answers *"would this passenger survive?"* with one
HTTP call:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"pclass": 3, "sex": "male", "age": 22, "fare": 7.25, "title": "Mr"}'
# -> {"survival_prediction": 0, "survival_probability": 0.076, ...}
```

## How to run

Each folder is self-contained. Download `titanic.csv` from the
[release](https://github.com/rabinarayanpatra350-biotech/titanic-ml-internship/releases/tag/v1.0),
place it in the week folder, then:

```bash
pip install -r week-1-preprocessing/requirements.txt   # (or the week's own requirements)
python week-1-preprocessing/preprocess.py

python week-2-supervised-models/train_models.py

python week-3-clustering-evaluation/cluster_and_evaluate.py

cd week-4-deployment-api
python train_and_serialize.py   # builds model.pkl
python app.py                   # API live at http://127.0.0.1:5000
```

## Original per-week repositories

Each week was also submitted as its own repository during the internship (these stay
live because the submission links point to them):

- Week 1: [titanic-data-preprocessing](https://github.com/rabinarayanpatra350-biotech/titanic-data-preprocessing)
- Week 2: [titanic-supervised-models](https://github.com/rabinarayanpatra350-biotech/titanic-supervised-models)
- Week 3: [titanic-clustering-evaluation](https://github.com/rabinarayanpatra350-biotech/titanic-clustering-evaluation)
- Week 4: [titanic-ml-api](https://github.com/rabinarayanpatra350-biotech/titanic-ml-api)

## Release archive

The [v1.0 release](https://github.com/rabinarayanpatra350-biotech/titanic-ml-internship/releases/tag/v1.0)
contains the full internship archive in one download: all four written reports (PDF),
the capstone presentation, the serialized model, the datasets, and every results table.

## Tech stack

Python 3.12 · pandas · NumPy · Matplotlib · scikit-learn · SciPy · joblib · Flask

---

*Educational internship portfolio — demonstrates the complete ML workflow:
preprocessing → model comparison → unsupervised analysis & evaluation → deployment.*
