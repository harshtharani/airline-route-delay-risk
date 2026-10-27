"""Train and evaluate the route risk models with a time-ordered split.

Setup: features come from month t, the label is the risk class in month t+1.
  - Training pairs: label month in TRAIN_MONTHS.  Test pairs: label month in TEST_MONTHS.
  - Risk cut-offs (terciles) are computed from TRAINING label rates only, then reused on the test month.
Models compared on the test month:
  1. persistence : "next month's risk = this month's risk" (no learning; the bar to beat)
  2. xgb_baseline: XGBoost on route, traffic and weather features (no graph features)
  3. xgb_graph   : same plus centrality metrics and GCN embeddings
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from build_graph import build_graph, airport_metrics
from config import (PROCESSED, OUTPUTS, TRAIN_MONTHS, TEST_MONTHS, RISK_QUANTILES, RISK_NAMES, SEED, check_consecutive)
from features import build_pair, feature_sets, load_weather
from gcn import train_gcn, link_auc


def fit_xgb(X, y):
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      objective="multi:softprob", eval_metric="mlogloss", random_state=SEED, n_jobs=-1, tree_method="hist")
    m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m


def scores(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": float(f.mean()),
            "per_class": {RISK_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i in range(3)},
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def bootstrap_ci(y, preds, B=1000):
    """95% bootstrap intervals for macro F1 and for the differences between models (resampling test routes)."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = rng.integers(0, n, (B, n))
    f = {k: np.array([f1_score(y[i], v[i], average="macro", labels=[0, 1, 2], zero_division=0) for i in idx])
         for k, v in preds.items()}
    q = lambda a: [float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {"macro_f1": {k: q(v) for k, v in f.items()}, "differences": {}}
    for a, b in (("xgb_graph", "xgb_baseline"), ("xgb_graph", "persistence"), ("xgb_baseline", "persistence")):
        mean, lo, hi = q(f[a] - f[b])
        out["differences"][f"{a} minus {b}"] = {"mean": mean, "low": lo, "high": hi, "includes_zero": bool(lo <= 0 <= hi)}
    return out


def shap_3d(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return np.asarray(sv)          # (rows, features, classes)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    stats = pd.read_csv(PROCESSED / "route_month_stats.csv")
    months = TRAIN_MONTHS + TEST_MONTHS
    check_consecutive(months)
    missing = [m for m in months if m not in set(stats.month)]
    if missing:
        raise SystemExit(f"Months {missing} not found in the data. Check config.py. Found: {sorted(stats.month.unique())}")
    if len(TRAIN_MONTHS) < 2:
        raise SystemExit("Need at least 2 TRAIN_MONTHS (one month of features and one of labels).")
    pairs = list(zip(months[:-1], months[1:]))
    train_pairs = [p for p in pairs if p[1] in TRAIN_MONTHS]
    test_pairs = [p for p in pairs if p[1] in TEST_MONTHS]
    print("train pairs (features -> label):", train_pairs, "| test pairs:", test_pairs)

    # 1) GCN trained only on the first training month's graph
    G0, _ = build_graph(stats, [months[0]])
    m0 = airport_metrics(G0)
    print(f"Training GCN on {months[0]} graph: {G0.number_of_nodes()} airports, {G0.number_of_edges()} routes")
    gcn = train_gcn(G0, m0)
    Gt, _ = build_graph(stats, [test_pairs[0][0]])
    auc = link_auc(gcn, Gt, airport_metrics(Gt))
    print(f"GCN sanity: link-prediction AUC on a later month's graph = {auc:.3f}")
    plt.figure(figsize=(5, 3)); plt.plot(gcn["losses"]); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("GCN training loss")
    plt.tight_layout(); plt.savefig(OUTPUTS / "gcn_loss.png", dpi=140); plt.close()

    # 2) Build tables
    weather = load_weather()
    train = pd.concat([build_pair(stats, f, l, gcn, weather) for f, l in train_pairs], ignore_index=True)
    test = pd.concat([build_pair(stats, f, l, gcn, weather) for f, l in test_pairs], ignore_index=True)

    # 3) Risk classes: cut-offs from TRAINING label rates only
    cuts = np.quantile(train.label_rate, RISK_QUANTILES)
    to_class = lambda r: np.digitize(r, cuts)
    ytr, yte = to_class(train.label_rate), to_class(test.label_rate)
    print("risk cut-offs (disruption rate):", np.round(cuts, 3))
    print("train class counts:", np.bincount(ytr, minlength=3), "| test class counts:", np.bincount(yte, minlength=3))

    base_cols, graph_cols = feature_sets(train)
    print(f"features: baseline {len(base_cols)}, graph model {len(graph_cols)}")
    m_base = fit_xgb(train[base_cols], ytr)
    m_graph = fit_xgb(train[graph_cols], ytr)

    preds = {"persistence": to_class(test.disrupted_rate),
             "xgb_baseline": m_base.predict(test[base_cols]),
             "xgb_graph": m_graph.predict(test[graph_cols])}
    results = {k: scores(yte, v) for k, v in preds.items()}
    comp = pd.DataFrame({k: {"accuracy": v["accuracy"], "balanced_accuracy": v["balanced_accuracy"], "macro_f1": v["macro_f1"],
                             "high_precision": v["per_class"]["high"]["precision"], "high_recall": v["per_class"]["high"]["recall"]}
                         for k, v in results.items()}).T.round(3)
    comp.to_csv(OUTPUTS / "model_comparison.csv")
    print("\nTest-month results:\n", comp.to_string())

    # 3b) Is any difference bigger than random noise? Bootstrap the test routes.
    boot = bootstrap_ci(yte, preds)
    print("\nBootstrap 95% intervals for macro F1 (resampling test routes):")
    for k, (mean, lo, hi) in boot["macro_f1"].items():
        print(f"  {k:13s} {mean:.3f}  [{lo:.3f}, {hi:.3f}]")
    for k, d in boot["differences"].items():
        verdict = "interval includes 0: NOT a clear difference" if d["includes_zero"] else "interval excludes 0: a real difference"
        print(f"  {k}: {d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]  -> {verdict}")

    # 4) Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (k, v) in zip(axes, results.items()):
        cm = np.array(v["confusion_matrix"]); ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{k}\nmacro F1 {v['macro_f1']:.2f}", fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels(RISK_NAMES); ax.set_yticks(range(3)); ax.set_yticklabels(RISK_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(OUTPUTS / "confusion_matrices.png", dpi=140); plt.close()

    # 5) SHAP explanations for the graph model
    sv = shap_3d(m_graph, test[graph_cols])
    pred_g = preds["xgb_graph"]
    n = len(test)
    own = sv[np.arange(n), :, pred_g]                          # SHAP values for each row's predicted class
    names = np.array(graph_cols)
    top = np.argsort(-np.abs(own), axis=1)[:, :3]
    drivers = [", ".join(f"{names[j]} ({own[i, j]:+.2f})" for j in top[i]) for i in range(n)]
    imp = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=graph_cols).sort_values(ascending=False)
    imp.to_csv(OUTPUTS / "shap_importance.csv", header=["mean_abs_shap"])
    plt.figure(figsize=(7, 5)); imp.head(15)[::-1].plot.barh(color="#1F3A5F")
    plt.xlabel("mean |SHAP| (average over classes)"); plt.title("What drives the graph model's predictions")
    plt.tight_layout(); plt.savefig(OUTPUTS / "shap_importance.png", dpi=140); plt.close()

    # 6) Save predictions and summary
    proba = m_graph.predict_proba(test[graph_cols])
    out = test[["Origin", "Dest", "feature_month", "label_month", "flights", "disrupted_rate", "label_rate"]].copy()
    out["true_class"] = [RISK_NAMES[i] for i in yte]
    for k, v in preds.items():
        out[f"pred_{k}"] = [RISK_NAMES[i] for i in v]
    for j, nme in enumerate(RISK_NAMES):
        out[f"p_{nme}"] = proba[:, j].round(4)
    out["top_drivers"] = drivers
    out.to_csv(OUTPUTS / "predictions_test.csv", index=False)
    m_graph.save_model(str(OUTPUTS / "xgb_graph.json"))
    summary = {"train_pairs": train_pairs, "test_pairs": test_pairs, "risk_cutoffs": cuts.tolist(),
               "train_rows": len(train), "test_rows": len(test), "baseline_features": base_cols, "graph_features": graph_cols,
               "gcn_link_auc_later_month": auc, "results": results, "bootstrap": boot}
    (OUTPUTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    gain = comp.loc["xgb_graph", "macro_f1"] - comp.loc["xgb_baseline", "macro_f1"]
    print(f"\nGraph features changed macro F1 by {gain:+.3f} vs the baseline XGBoost. Report this honestly either way.")
    print("Saved results to", OUTPUTS)


if __name__ == "__main__":
    main()
