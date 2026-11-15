#!/usr/bin/env python3
"""Runs the experiments in the README and redraws figures/. Takes ~20 min, mostly the SVM."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.models.classical import (
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
)

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.25
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
METRICS_PATH = PROJECT_ROOT / "results" / "paper_metrics.json"

# same palette as config/config.py so the figures match the rest of the project
C_BLUE = "#2E86AB"
C_PURPLE = "#A23B72"
C_GREEN = "#3D8C40"
C_ORANGE = "#F18F01"
C_RED = "#C73E1D"
C_GREY = "#6E6E6E"
MODEL_COLORS = {
    "SVM": C_BLUE,
    "LogisticRegression": C_PURPLE,
    "RandomForest": C_GREEN,
    "XGBoost": C_ORANGE,
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)

RESULTS: dict = {}


def model_factory() -> dict:
    # fresh instances every time, nothing carries over between folds
    return {
        "RandomForest": RandomForestModel(),
        "LogisticRegression": LogisticRegressionModel(),
        "SVM": SVMModel(),
        "XGBoost": XGBoostModel(),
    }


def score(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def bootstrap_ci(y_true, y_pred, y_prob, metric: str, n: int = 1000) -> tuple:
    """Percentile bootstrap 95% CI, resampling the test set."""
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(score(y_true[idx], y_pred[idx], y_prob[idx])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
print("[1/10] Loading data and engineering features")
df_raw = DataLoader().load_data()
groups_all = df_raw["host_species"].to_numpy()

engineer = FeatureEngineer()
df_eng = engineer.fit_transform(df_raw)
X_all = df_eng.drop(columns=["infection_status"])
y_all = df_eng["infection_status"].to_numpy()

RESULTS["dataset"] = {
    "n_records": int(len(df_raw)),
    "n_species": int(df_raw["host_species"].nunique()),
    "n_families": int(df_raw["host_family"].nunique()),
    "n_orders": int(df_raw["host_order"].nunique()),
    "n_raw_columns": int(df_raw.shape[1]),
    "n_engineered_features": int(X_all.shape[1]),
    "records_per_species": int(df_raw["host_species"].value_counts().iloc[0]),
    "infection_rate": float(df_raw["infection_status"].mean()),
    "n_positive": int(df_raw["infection_status"].sum()),
    "missing_values": int(df_raw.isnull().sum().sum()),
    "viruses": df_raw["virus"].value_counts().to_dict(),
    "infection_rate_by_order": df_raw.groupby("host_order")["infection_status"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
    .to_dict("index"),
    "infection_rate_by_species": df_raw.groupby("host_species")["infection_status"]
    .mean()
    .sort_values(ascending=False)
    .to_dict(),
}

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
)
print(f"      train={len(X_train)}  test={len(X_test)}  features={X_all.shape[1]}")

# --------------------------------------------------------------------------
# Experiment 1: held-out model comparison (the pipeline's headline numbers)
# --------------------------------------------------------------------------
print("[2/10] Experiment 1: held-out model comparison")
holdout: dict = {}
curves: dict = {}
for name, model in model_factory().items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    m = score(y_test, y_pred, y_prob)
    m["f1_ci"] = bootstrap_ci(y_test, y_pred, y_prob, "f1")
    m["roc_auc_ci"] = bootstrap_ci(y_test, y_pred, y_prob, "roc_auc")
    holdout[name] = m
    curves[name] = {"prob": y_prob, "pred": y_pred}
    print(f"      {name:<20} F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}")

BEST = max(holdout, key=lambda k: holdout[k]["f1"])
RESULTS["experiment_1_holdout"] = holdout
RESULTS["best_model"] = BEST
print(f"      best by F1: {BEST}")

# --------------------------------------------------------------------------
# Experiment 2: naive baselines
# --------------------------------------------------------------------------
print("[3/10] Experiment 2: naive baselines")
rng = np.random.default_rng(SEED)
prevalence = float(y_train.mean())
baselines = {
    "MajorityClass (always 0)": (
        np.zeros(len(y_test), dtype=int),
        np.full(len(y_test), prevalence),
    ),
    "AlwaysPositive (always 1)": (
        np.ones(len(y_test), dtype=int),
        np.full(len(y_test), prevalence),
    ),
    "Random (prevalence-matched)": (
        (rng.random(len(y_test)) < prevalence).astype(int),
        rng.random(len(y_test)),
    ),
}
RESULTS["experiment_2_baselines"] = {
    k: score(y_test, p, pr) for k, (p, pr) in baselines.items()
}
for k, v in RESULTS["experiment_2_baselines"].items():
    print(f"      {k:<30} F1={v['f1']:.3f}  AUC={v['roc_auc']:.3f}")

# --------------------------------------------------------------------------
# Experiment 3: record-level vs species-grouped cross-validation
# --------------------------------------------------------------------------
print("[4/10] Experiment 3: record-level vs species-grouped CV (this is slow)")
cv_results: dict = {}
for name in model_factory():
    record_f1, record_auc, group_f1, group_auc = [], [], [], []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X_all, y_all):
        m = model_factory()[name]
        m.fit(X_all.iloc[tr], y_all[tr])
        p = m.predict(X_all.iloc[te])
        pr = m.predict_proba(X_all.iloc[te])[:, 1]
        record_f1.append(f1_score(y_all[te], p, zero_division=0))
        record_auc.append(roc_auc_score(y_all[te], pr))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in sgkf.split(X_all, y_all, groups=groups_all):
        m = model_factory()[name]
        m.fit(X_all.iloc[tr], y_all[tr])
        p = m.predict(X_all.iloc[te])
        pr = m.predict_proba(X_all.iloc[te])[:, 1]
        group_f1.append(f1_score(y_all[te], p, zero_division=0))
        group_auc.append(roc_auc_score(y_all[te], pr))

    cv_results[name] = {
        "record_cv_f1_mean": float(np.mean(record_f1)),
        "record_cv_f1_std": float(np.std(record_f1)),
        "record_cv_auc_mean": float(np.mean(record_auc)),
        "record_cv_auc_std": float(np.std(record_auc)),
        "species_cv_f1_mean": float(np.mean(group_f1)),
        "species_cv_f1_std": float(np.std(group_f1)),
        "species_cv_auc_mean": float(np.mean(group_auc)),
        "species_cv_auc_std": float(np.std(group_auc)),
    }
    r = cv_results[name]
    print(
        f"      {name:<20} record AUC={r['record_cv_auc_mean']:.3f}"
        f"  species-grouped AUC={r['species_cv_auc_mean']:.3f}"
    )
RESULTS["experiment_3_cross_validation"] = cv_results

# --------------------------------------------------------------------------
# Experiment 4: feature-group ablation (best model, held-out split)
# --------------------------------------------------------------------------
print("[5/10] Experiment 4: feature-group ablation")
GROUPS = {
    "Viral genome": [
        "viral_genome_length",
        "gc_content",
        "gc_content_squared",
        "gc_content_log",
        "cpg_frequency",
        "cpg_density",
        "cpg_log",
        "glycoprotein_length",
        "gp_hydrophobicity",
        "gp_hydro_abs",
    ],
    "Host life history": [
        "body_mass_kg",
        "body_mass_log",
        "body_mass_sqrt",
        "lifespan_years",
        "lifespan_log",
        "geographic_range_km2",
        "range_log",
    ],
    "Taxonomy": [
        "host_species",
        "host_family",
        "host_order",
        "is_primate",
        "is_chiroptera",
        "is_rodentia",
        "is_artiodactyla",
        "is_hominidae",
        "is_pteropodidae",
        "is_vespertilionidae",
        "is_rhinolophidae",
        "is_muridae",
        "taxonomic_risk",
    ],
    "Ecology & behaviour": [
        "diet_type",
        "social_behavior",
        "habitat_type",
        "human_contact_frequency",
        "human_contact_score",
        "social_group_score",
        "urban_habitat",
        "forest_habitat",
        "cave_habitat",
        "farm_habitat",
        "is_frugivore",
        "is_omnivore",
        "is_herbivore",
        "is_insectivore",
        "ecological_risk",
        "contact_risk",
    ],
    "Derived risk scores": [
        "bdbv_risk",
        "baseline_risk",
        "combined_risk",
        "high_risk",
    ],
}
full_ref = holdout[BEST]
ablation = {"Full feature set": {"f1": full_ref["f1"], "roc_auc": full_ref["roc_auc"]}}
for gname, cols in GROUPS.items():
    present = [c for c in cols if c in X_train.columns]
    m = model_factory()[BEST]
    m.fit(X_train.drop(columns=present), y_train)
    Xt = X_test.drop(columns=present)
    p, pr = m.predict(Xt), m.predict_proba(Xt)[:, 1]
    ablation[f"- {gname}"] = {
        "f1": f1_score(y_test, p, zero_division=0),
        "roc_auc": roc_auc_score(y_test, pr),
        "n_features_removed": len(present),
    }
    print(
        f"      drop {gname:<22} ({len(present):>2} feats)"
        f"  F1={ablation[f'- {gname}']['f1']:.3f}"
        f"  AUC={ablation[f'- {gname}']['roc_auc']:.3f}"
    )
RESULTS["experiment_4_ablation"] = ablation

# --------------------------------------------------------------------------
# Experiment 5: permutation importance for the selected model
# --------------------------------------------------------------------------
print("[6/10] Experiment 5: permutation importance")
best_model = model_factory()[BEST]
best_model.fit(X_train, y_train)


class _SklearnAdapter:
    """Expose the project's model wrapper to sklearn's permutation_importance."""

    def __init__(self, m):
        self.m = m

    def fit(self, X, y):  # pragma: no cover - required by the sklearn API
        return self

    def predict(self, X):
        return self.m.predict(pd.DataFrame(X, columns=X_train.columns))

    def score(self, X, y):
        return roc_auc_score(
            y, self.m.predict_proba(pd.DataFrame(X, columns=X_train.columns))[:, 1]
        )


sub = min(1500, len(X_test))
perm = permutation_importance(
    _SklearnAdapter(best_model),
    X_test.iloc[:sub].to_numpy(),
    y_test[:sub],
    n_repeats=5,
    random_state=SEED,
    scoring=None,
)
imp = (
    pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": perm.importances_mean,
            "std": perm.importances_std,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
RESULTS["experiment_5_permutation_importance"] = imp.head(20).to_dict("records")
print("      top 5:", ", ".join(imp.head(5)["feature"]))

# --------------------------------------------------------------------------
# Experiment 6: per-species error analysis for the selected model
# --------------------------------------------------------------------------
print("[7/10] Experiment 6: per-species performance")
test_species = df_raw.loc[X_test.index, "host_species"]
per_species = (
    pd.DataFrame(
        {
            "species": test_species.to_numpy(),
            "y": y_test,
            "pred": curves[BEST]["pred"],
            "prob": curves[BEST]["prob"],
        }
    )
    .groupby("species")
    .agg(
        n=("y", "size"),
        true_rate=("y", "mean"),
        pred_rate=("pred", "mean"),
        mean_prob=("prob", "mean"),
        accuracy=("y", lambda s: None),
    )
)
per_species["accuracy"] = (
    pd.DataFrame({"species": test_species.to_numpy(), "ok": y_test == curves[BEST]["pred"]})
    .groupby("species")["ok"]
    .mean()
)
per_species = per_species.sort_values("true_rate", ascending=False)
RESULTS["experiment_6_per_species"] = per_species.round(4).to_dict("index")

# --------------------------------------------------------------------------
# Experiment 7: how much signal is species identity alone?
# --------------------------------------------------------------------------
print("[8/10] Experiment 7: species-identity oracle")
train_species = df_raw.loc[X_train.index, "host_species"]
species_rate = (
    pd.DataFrame({"s": train_species.to_numpy(), "y": y_train}).groupby("s")["y"].mean()
)
oracle_score = test_species.map(species_rate).to_numpy()
order_rate = (
    pd.DataFrame({"o": df_raw.loc[X_train.index, "host_order"].to_numpy(), "y": y_train})
    .groupby("o")["y"]
    .mean()
)
order_score = df_raw.loc[X_test.index, "host_order"].map(order_rate).to_numpy()
thresholds = np.arange(0.05, 0.90, 0.01)
oracle_f1, oracle_t = max(
    (f1_score(y_test, (oracle_score > t).astype(int), zero_division=0), t)
    for t in thresholds
)
RESULTS["experiment_7_identity_oracle"] = {
    "species_rate_oracle_auc": float(roc_auc_score(y_test, oracle_score)),
    "species_rate_oracle_best_f1": float(oracle_f1),
    "species_rate_oracle_best_threshold": float(oracle_t),
    "order_rate_oracle_auc": float(roc_auc_score(y_test, order_score)),
    "selected_model_auc": float(holdout[BEST]["roc_auc"]),
    "fraction_of_oracle_auc_gain_recovered": float(
        (holdout[BEST]["roc_auc"] - 0.5) / (roc_auc_score(y_test, oracle_score) - 0.5)
    ),
}
print(
    f"      species oracle AUC={RESULTS['experiment_7_identity_oracle']['species_rate_oracle_auc']:.3f}"
    f"  order oracle AUC={RESULTS['experiment_7_identity_oracle']['order_rate_oracle_auc']:.3f}"
)

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
print("[9/10] Rendering figures")

# Figure 1 - dataset composition
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
counts = df_raw["infection_status"].value_counts().sort_index()
axes[0].bar(
    ["Resistant (0)", "Susceptible (1)"],
    counts.values,
    color=[C_GREY, C_RED],
    width=0.6,
)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 120, f"{v:,}\n({v/len(df_raw):.1%})", ha="center", fontsize=9)
axes[0].set_ylim(0, len(df_raw) * 0.92)
axes[0].set_ylabel("Records")
axes[0].set_title("A  Class balance (n = 10,000)")

by_order = (
    df_raw.groupby("host_order")["infection_status"].agg(["mean", "count"]).sort_values("mean")
)
axes[1].barh(by_order.index, by_order["mean"], color=C_BLUE)
axes[1].axvline(
    df_raw["infection_status"].mean(), color=C_RED, ls="--", lw=1.2, label="Overall 24.1%"
)
for i, (m, c) in enumerate(zip(by_order["mean"], by_order["count"])):
    axes[1].text(0.006, i, f"n={c:,}", va="center", fontsize=7.5, color="white")
axes[1].set_xlim(0, by_order["mean"].max() * 1.15)
axes[1].set_xlabel("Infection rate")
axes[1].set_title("B  Infection rate by host order")
axes[1].legend(loc="lower right", fontsize=8)

top_sp = df_raw.groupby("host_species")["infection_status"].mean().sort_values().tail(12)
axes[2].barh(
    [s.replace("_", " ") for s in top_sp.index],
    top_sp.values,
    color=C_PURPLE,
)
axes[2].set_xlim(0, top_sp.max() * 1.12)
axes[2].set_xlabel("Infection rate")
axes[2].set_title("C  Highest-rate species (top 12)")
axes[2].tick_params(axis="y", labelsize=8)
for i, v in enumerate(top_sp.values):
    axes[2].text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=7.5, color=C_GREY)
fig.suptitle("Figure 1. Composition of the simulated BDBV host dataset", y=1.03, fontsize=13)
fig.savefig(FIG_DIR / "fig01_dataset_composition.png")
plt.close(fig)

# Figure 2 - pipeline diagram
fig, ax = plt.subplots(figsize=(13, 3.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 26)
ax.axis("off")
stages = [
    ("Raw table\n10,000 x 18\n41 species", C_BLUE),
    ("FeatureEngineer\nfit_transform\n18 -> 51 features", C_PURPLE),
    ("Stratified split\n7,500 train\n2,500 test", C_GREEN),
    ("4 classifiers\nRF / LR / SVM / XGB\nclass_weight=balanced", C_ORANGE),
    ("Select by F1\n-> SVM", C_RED),
    ("Pickled model\n+ feature engineer\nFlask API", C_GREY),
]
w, gap = 13.5, 3.2
for i, (label, color) in enumerate(stages):
    x = 2 + i * (w + gap)
    ax.add_patch(
        plt.Rectangle(
            (x, 7), w, 12, facecolor=color, alpha=0.14, edgecolor=color, lw=1.6, zorder=2
        )
    )
    ax.text(x + w / 2, 13, label, ha="center", va="center", fontsize=8.5, zorder=3)
    if i < len(stages) - 1:
        ax.annotate(
            "",
            xy=(x + w + gap - 0.4, 13),
            xytext=(x + w + 0.4, 13),
            arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=1.4),
        )
ax.text(
    50,
    3.2,
    "Feature engineering is fitted on all 10,000 records before the split, and records of the same species appear in both sides of it.\n"
    "Section 'Limitations' quantifies what both choices cost.",
    ha="center",
    fontsize=8,
    color=C_RED,
    style="italic",
)
ax.set_title("Figure 2. Training and deployment pipeline as implemented", fontsize=13, pad=4)
fig.savefig(FIG_DIR / "fig02_pipeline.png")
plt.close(fig)

# Figure 3 - model comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
names = list(holdout)
xpos = np.arange(len(names))
metrics_to_plot = [("accuracy", C_GREY), ("f1", C_BLUE), ("roc_auc", C_PURPLE)]
width = 0.26
for j, (metric, color) in enumerate(metrics_to_plot):
    vals = [holdout[n][metric] for n in names]
    axes[0].bar(xpos + (j - 1) * width, vals, width, label=metric.upper(), color=color)
    for xi, v in zip(xpos + (j - 1) * width, vals):
        axes[0].text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(names, rotation=12, fontsize=8.5)
axes[0].set_ylim(0, 1.0)
axes[0].set_title("A  Held-out test metrics (n = 2,500)")
axes[0].legend(fontsize=8, ncol=3)

for i, n in enumerate(names):
    lo, hi = holdout[n]["f1_ci"]
    axes[1].errorbar(
        holdout[n]["f1"],
        i,
        xerr=[[holdout[n]["f1"] - lo], [hi - holdout[n]["f1"]]],
        fmt="o",
        color=MODEL_COLORS[n],
        capsize=4,
        markersize=7,
    )
    axes[1].text(hi + 0.012, i, f"{holdout[n]['f1']:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=8)
axes[1].set_yticks(range(len(names)))
axes[1].set_yticklabels(names, fontsize=8.5)
axes[1].set_xlim(0, 0.75)
axes[1].set_xlabel("F1 score")
axes[1].set_title("B  F1 with 95% bootstrap CI (1,000 resamples)")
fig.suptitle("Figure 3. Model comparison on the held-out test set", y=1.02, fontsize=13)
fig.savefig(FIG_DIR / "fig03_model_comparison.png")
plt.close(fig)

# Figure 4 - ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for n in names:
    fpr, tpr, _ = roc_curve(y_test, curves[n]["prob"])
    axes[0].plot(fpr, tpr, color=MODEL_COLORS[n], lw=1.7, label=f"{n} (AUC {holdout[n]['roc_auc']:.3f})")
    prec, rec, _ = precision_recall_curve(y_test, curves[n]["prob"])
    axes[1].plot(rec, prec, color=MODEL_COLORS[n], lw=1.7, label=f"{n} (AP {holdout[n]['pr_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", color=C_GREY, lw=1, label="Chance")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("A  ROC curves")
axes[0].legend(fontsize=8, loc="lower right")
axes[1].axhline(y_test.mean(), ls="--", color=C_GREY, lw=1, label=f"Prevalence {y_test.mean():.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("B  Precision-recall curves")
axes[1].legend(fontsize=8, loc="upper right")
fig.suptitle("Figure 4. Discrimination on the held-out test set", y=1.02, fontsize=13)
fig.savefig(FIG_DIR / "fig04_roc_pr_curves.png")
plt.close(fig)

# Figure 5 - confusion matrix + calibration for the selected model
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
cm = confusion_matrix(y_test, curves[BEST]["pred"])
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1], ["Pred 0", "Pred 1"])
axes[0].set_yticks([0, 1], ["True 0", "True 1"])
axes[0].grid(False)
for (i, j), v in np.ndenumerate(cm):
    axes[0].text(
        j,
        i,
        f"{v:,}\n({v/cm.sum():.1%})",
        ha="center",
        va="center",
        fontsize=11,
        color="white" if v > cm.max() * 0.55 else "black",
    )
axes[0].set_title(f"A  Confusion matrix - {BEST}")
fig.colorbar(im, ax=axes[0], fraction=0.045)

frac_pos, mean_pred = calibration_curve(y_test, curves[BEST]["prob"], n_bins=10)
axes[1].plot([0, 1], [0, 1], ls="--", color=C_GREY, lw=1, label="Perfect calibration")
axes[1].plot(mean_pred, frac_pos, "o-", color=C_BLUE, lw=1.7, label=BEST)
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Observed frequency")
axes[1].set_title("B  Calibration (10 bins)")
axes[1].legend(fontsize=8)
fig.suptitle(f"Figure 5. Error structure of the selected model ({BEST})", y=1.02, fontsize=13)
fig.savefig(FIG_DIR / "fig05_confusion_calibration.png")
plt.close(fig)

RESULTS["confusion_matrix_best"] = {
    "tn": int(cm[0, 0]),
    "fp": int(cm[0, 1]),
    "fn": int(cm[1, 0]),
    "tp": int(cm[1, 1]),
}
RESULTS["calibration_best"] = {
    "mean_predicted": [float(v) for v in mean_pred],
    "observed_frequency": [float(v) for v in frac_pos],
}

# Figure 6 - record-level vs species-grouped CV
fig, ax = plt.subplots(figsize=(9, 4.4))
xpos = np.arange(len(names))
rec = [cv_results[n]["record_cv_auc_mean"] for n in names]
rec_e = [cv_results[n]["record_cv_auc_std"] for n in names]
grp = [cv_results[n]["species_cv_auc_mean"] for n in names]
grp_e = [cv_results[n]["species_cv_auc_std"] for n in names]
ax.bar(xpos - 0.2, rec, 0.4, yerr=rec_e, capsize=4, color=C_BLUE, label="Record-level 5-fold CV")
ax.bar(xpos + 0.2, grp, 0.4, yerr=grp_e, capsize=4, color=C_RED, label="Species-grouped 5-fold CV")
for xi, v, e in zip(xpos - 0.2, rec, rec_e):
    ax.text(xi, v + e + 0.02, f"{v:.3f}", ha="center", fontsize=8)
for xi, v, e in zip(xpos + 0.2, grp, grp_e):
    ax.text(xi, v + e + 0.02, f"{v:.3f}", ha="center", fontsize=8)
ax.axhline(0.5, ls="--", color=C_GREY, lw=1, label="Chance (AUC 0.5)")
ax.axhline(
    RESULTS["experiment_7_identity_oracle"]["species_rate_oracle_auc"],
    ls=":",
    color=C_GREEN,
    lw=1.6,
    label="Species-identity oracle ("
    f"{RESULTS['experiment_7_identity_oracle']['species_rate_oracle_auc']:.3f})",
)
ax.set_xticks(xpos)
ax.set_xticklabels(names, rotation=12, fontsize=9)
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0, 0.95)
ax.legend(fontsize=8)
ax.set_title(
    "Figure 6. What happens when no record of a test species is seen in training",
    fontsize=12,
)
fig.savefig(FIG_DIR / "fig06_species_grouped_cv.png")
plt.close(fig)

# Figure 7 - ablation
fig, ax = plt.subplots(figsize=(9, 4.4))
labels = list(ablation)
f1s = [ablation[k]["f1"] for k in labels]
aucs = [ablation[k]["roc_auc"] for k in labels]
xpos = np.arange(len(labels))
ax.bar(xpos - 0.2, f1s, 0.4, color=C_BLUE, label="F1")
ax.bar(xpos + 0.2, aucs, 0.4, color=C_PURPLE, label="ROC-AUC")
ax.axhline(ablation["Full feature set"]["roc_auc"], ls="--", color=C_GREY, lw=1)
for xi, v in zip(xpos - 0.2, f1s):
    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
for xi, v in zip(xpos + 0.2, aucs):
    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(xpos)
ax.set_xticklabels([l.replace("- ", "drop\n") for l in labels], fontsize=8)
ax.set_ylim(0, 0.85)
ax.legend(fontsize=8)
ax.set_title(f"Figure 7. Feature-group ablation ({BEST}, held-out test set)", fontsize=12)
fig.savefig(FIG_DIR / "fig07_ablation.png")
plt.close(fig)

# Figure 8 - permutation importance
fig, ax = plt.subplots(figsize=(8.5, 5.4))
top = imp.head(15).iloc[::-1]
ax.barh(
    [f.replace("_", " ") for f in top["feature"]],
    top["importance"],
    xerr=top["std"],
    color=C_BLUE,
    capsize=3,
)
ax.axvline(0, color=C_GREY, lw=1)
ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
ax.tick_params(axis="y", labelsize=8.5)
ax.set_title(f"Figure 8. Permutation importance ({BEST}, 5 repeats)", fontsize=12)
fig.savefig(FIG_DIR / "fig08_permutation_importance.png")
plt.close(fig)

# Figure 9 - per-species predicted vs observed
fig, ax = plt.subplots(figsize=(7.5, 6))
ps = per_species.reset_index()
ax.scatter(ps["true_rate"], ps["mean_prob"], s=38, color=C_BLUE, alpha=0.8, edgecolor="white")
lims = [0, max(ps["true_rate"].max(), ps["mean_prob"].max()) * 1.12]
ax.plot(lims, lims, ls="--", color=C_GREY, lw=1, label="y = x")
for _, r in ps.iterrows():
    if r["true_rate"] > 0.35 or r["mean_prob"] > 0.45 or r["true_rate"] < 0.06:
        ax.annotate(
            r["species"].replace("_", " "),
            (r["true_rate"], r["mean_prob"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
ax.set_xlabel("Observed infection rate in the test set")
ax.set_ylabel("Mean predicted probability")
ax.set_xlim(lims)
ax.legend(fontsize=8)
ax.set_title(f"Figure 9. Per-species predicted vs observed rate ({BEST})", fontsize=12)
fig.savefig(FIG_DIR / "fig09_per_species.png")
plt.close(fig)

# --------------------------------------------------------------------------
print("[10/10] Writing results/paper_metrics.json")
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


METRICS_PATH.write_text(json.dumps(_clean(RESULTS), indent=2), encoding="utf-8")
print(f"Done. Figures in {FIG_DIR}, metrics in {METRICS_PATH}")
