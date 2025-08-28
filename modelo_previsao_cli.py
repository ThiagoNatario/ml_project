import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, average_precision_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

# ----------------------
# Args
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", required=True, help="Caminho para bank.csv (separador ;)")
parser.add_argument("--target", default="y")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

outdir = Path("outputs")
outdir.mkdir(parents=True, exist_ok=True)

# ----------------------
# Load & clean
# ----------------------
df = pd.read_csv(args.csv_path, sep=";")
print(df.columns.tolist())
print(f"Dataset original: {len(df)} linhas")

# remover colunas de baixo valor ou indevidas para previsão
drop_cols = [c for c in ["duration", "day", "month", "poutcome"] if c in df.columns]
df_copy = df.drop(columns=drop_cols)

# unknown -> NaN e drop de linhas incompletas
df_copy = df_copy.replace("unknown", np.nan)
df_clean = df_copy.dropna()
print(f"Linhas após remoção de NaN/unknown: {len(df_clean)}")

# ----------------------
# Split X/y
# ----------------------
if args.target not in df_clean.columns:
    raise ValueError(f"Coluna alvo '{args.target}' não existe no dataset.")
X = df_clean.drop(columns=[args.target])
y = (df_clean[args.target].astype(str)
     .str.strip()
     .replace({'yes': 1, 'no': 0, 'y': 1, 'n': 0})).astype(int)

# ----------------------
# Preprocess
# ----------------------
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# ----------------------
# Estimadores
# ----------------------
lsvc = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid")
lr = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
gb = GradientBoostingClassifier()
xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss")

estimators = {
    "linearsvm": (lsvc, {
        "clf__estimator__C": [0.1, 1.0, 10.0],
        "clf__estimator__penalty": ["l2"],
        "clf__estimator__class_weight": ["balanced", None]
    }),
    "logreg": (lr, {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__class_weight": ["balanced", None]
    }),
    "gboost": (gb, {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 5],
        "clf__subsample": [0.8, 1.0]
    }),
    "xgboost": (xgb, {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 5]
    })
}

# ----------------------
# Train / eval
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.seed, stratify=y
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

best_models = {}
metrics_out = {}

for name, (estimator, param_dist) in estimators.items():
    pipe = Pipeline([("preprocess", preprocess), ("clf", estimator)])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=20, cv=cv,
        scoring="recall", n_jobs=-1, verbose=1, random_state=args.seed
    )
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_

    # Probabilidades (ou scores normalizados)
    if hasattr(search.best_estimator_.named_steps["clf"], "predict_proba"):
        y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    else:
        scores = search.best_estimator_.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    y_pred = search.best_estimator_.predict(X_test)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics_out[name] = {
        "recall": float(recall),
        "precision": float(precision),
        "roc_auc": float(roc),
        "average_precision": float(ap),
        "cv_best_params": search.best_params_,
        "classification_report": report
    }
    print(f"[{name}] RECALL={recall:.4f} | PRECISION={precision:.4f} | ROC AUC={roc:.4f} | AP={ap:.4f}")

# vencedor por recall
best_name = max(metrics_out, key=lambda k: metrics_out[k]["recall"])
best_model = best_models[best_name]
print(f"Melhor por RECALL: {best_name} ({metrics_out[best_name]['recall']:.4f})")

# figuras apenas do melhor
if hasattr(best_model.named_steps["clf"], "predict_proba"):
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
else:
    scores = best_model.decision_function(X_test)
    y_proba_best = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
y_pred_best = best_model.predict(X_test)

roc_disp = RocCurveDisplay.from_predictions(y_test, y_proba_best)
roc_disp.figure_.savefig(outdir / f"roc_{best_name}.png", bbox_inches="tight")
plt.close(roc_disp.figure_)

pr_disp = PrecisionRecallDisplay.from_predictions(y_test, y_proba_best)
pr_disp.figure_.savefig(outdir / f"pr_{best_name}.png", bbox_inches="tight")
plt.close(pr_disp.figure_)

cm_disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best)
cm_disp.figure_.savefig(outdir / f"cm_{best_name}.png", bbox_inches="tight")
plt.close(cm_disp.figure_)

# salvar artefatos
dump(best_model, outdir / "best_model.joblib")
dump(best_model.named_steps["preprocess"], outdir / "preprocess.joblib")

with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"metrics": metrics_out, "winner": best_name}, f, ensure_ascii=False, indent=2)

print("Concluído. Artefatos em 'outputs/'.")
