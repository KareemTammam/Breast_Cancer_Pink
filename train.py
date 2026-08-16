"""
Training script for the Breast Cancer Diagnosis Prediction project.

This reproduces the EXACT preprocessing pipeline, feature selection, and
train/test split from the original notebook (Breast_Cancer_Project.ipynb).
Hyperparameters used here are the best ones already found in the notebook's
GridSearchCV runs, so results match the original analysis without re-running
the (slow) grid search.

One correction vs. the notebook: the notebook's final `joblib.dump(model, ...)`
accidentally saved the Random Forest (because later boosting models were
assigned to differently-named variables and never reassigned `model`), even
though SVC scored highest on every metric. This script instead selects the
best model automatically by F1 score, same as a real deployment should.

Run once: `python train.py`
Produces everything under model/ that the Streamlit app needs.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/breast-cancer.csv")
df = df.drop(columns=["id"])

# ---------------------------------------------------------------------------
# 2. Preprocessing (identical to the notebook)
# ---------------------------------------------------------------------------
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])  # B=0, M=1

# Correlation-based feature selection (threshold > 0.20), same as the notebook
corr = df.corr(numeric_only=True)
selected_features = corr["diagnosis"][corr["diagnosis"] > 0.20]
selected_columns = selected_features.index.tolist()
selected_columns.remove("diagnosis")

X = df[selected_columns]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Models, using the best hyperparameters already found via GridSearchCV
#    in the original notebook.
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(C=10, gamma=0.01, kernel="rbf", random_state=RANDOM_STATE, probability=True),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        criterion="entropy",
        max_depth=6,
        min_samples_leaf=1,
        min_samples_split=5,
        class_weight=None,
    ),
    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
    ),
    "AdaBoost": AdaBoostClassifier(
        random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.5
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.05, max_depth=3
    ),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
}

results = []
confusion_matrices = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        }
    )
    confusion_matrices[name] = confusion_matrix(y_test, pred).tolist()
    trained_models[name] = model
    print(f"Trained {name}")

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------------------
# 4. Feature importance for the best model (native importance if available,
#    permutation importance fallback for kernel SVMs and similar).
# ---------------------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": selected_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
    fi_df = (
        pd.DataFrame({"feature": selected_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
else:
    r = permutation_importance(
        best_model, X_test_scaled, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    fi_df = (
        pd.DataFrame({"feature": selected_columns, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# 5. Target correlation (for EDA page) - on the selected raw features
# ---------------------------------------------------------------------------
target_corr = corr["diagnosis"].drop("diagnosis").sort_values(key=abs, ascending=False)

# ---------------------------------------------------------------------------
# 6. Persist everything the Streamlit app needs
# ---------------------------------------------------------------------------
with open("model/trained_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/feature_names.json", "w") as f:
    json.dump(selected_columns, f)

results_df.to_json("model/model_comparison.json", orient="records")
fi_df.to_json("model/feature_importance.json", orient="records")

with open("model/confusion_matrices.json", "w") as f:
    json.dump(confusion_matrices, f)

meta = {
    "best_model_name": best_model_name,
    "n_samples": int(df.shape[0]),
    "n_features_raw": int(X.shape[1]),
    "n_features_encoded": int(X.shape[1]),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "class_balance": y.value_counts(normalize=True).round(4).to_dict(),
    "class_counts": y.value_counts().to_dict(),
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f)

target_corr.to_json("model/target_correlation.json")

# Per-feature stats (min/max/mean) used to build sensible defaults & ranges
# for the numeric prediction-form inputs.
feature_stats = {
    col: {
        "min": float(X[col].min()),
        "max": float(X[col].max()),
        "mean": float(X[col].mean()),
    }
    for col in selected_columns
}
with open("model/feature_stats.json", "w") as f:
    json.dump(feature_stats, f)

# Small sample of raw data for the dataset overview page (original labels)
sample_df = df.copy()
sample_df["diagnosis"] = label_encoder.inverse_transform(sample_df["diagnosis"])
sample_df.sample(min(200, len(sample_df)), random_state=RANDOM_STATE).to_csv(
    "model/sample_data.csv", index=False
)

print("\nAll artifacts saved to model/")
