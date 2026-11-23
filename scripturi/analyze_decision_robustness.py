from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "rezultate" / "runs" / "L30_H30"
DEST = ROOT / "rezultate" / "reviewer4_robustness"
MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting", "LSTM", "RF-LSTM"]
SEEDS = [13, 37, 73]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
ALPHAS = np.round(np.arange(0.1, 0.91, 0.1), 2)
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load(seed, model):
    path = RUNS / f"seed_{seed}" / (model.replace(" ", "_") + ".npz")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\nRun the main benchmark first; this analysis never trains models."
        )
    with np.load(path) as d:
        yv = np.asarray(d["validation_y"]).astype(int).ravel()
        pv = np.asarray(d["validation_prob"]).astype(float).ravel()
        yt = np.asarray(d["test_y"]).astype(int).ravel()
        pt = np.asarray(d["test_prob"]).astype(float).ravel()
    for y, p in [(yv, pv), (yt, pt)]:
        if len(y) != len(p) or not np.isfinite(p).all():
            raise ValueError(f"Invalid predictions for {model}, seed {seed}")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Nonbinary labels")
    return yv, pv, yt, pt


def counts(y, p, t):
    pred = p >= t
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))
    return tn, fp, fn, tp


def stats(y, p, t):
    tn, fp, fn, tp = counts(y, p, t)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return sensitivity, specificity, fp, fn


def select_on_validation(y, p, alpha):
    # Match the benchmark's original threshold grid and tie-breaking.
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = []
    for t in grid:
        se, sp, _, _ = stats(y, p, t)
        scores.append(alpha * se + (1 - alpha) * sp)
    return float(grid[int(np.nanargmax(scores))])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = {}
    for seed in SEEDS:
        for model in MODELS:
            records[seed, model] = load(seed, model)

    # AUPRC is threshold-free. Keep seed-level observations distinct.
    apr = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, model in enumerate(MODELS):
        values = []
        for seed in SEEDS:
            _, _, y, p = records[seed, model]
            score = average_precision_score(y, p)
            values.append(score)
            apr.append(dict(seed=seed, model=model, auprc=score,
                            n_test_windows=len(y), positive_prevalence=float(y.mean())))
        # One held-out seed shown as an illustrative curve; no cross-seed concatenation.
        _, _, y, p = records[SEEDS[0], model]
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=COLORS[i % len(COLORS)],
                label=f"{model} (seed 13 AP={values[0]:.3f})", lw=1.8)
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Held-out precision–recall curves (seed 13, L30/H30)",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(DEST / "figure_R4_precision_recall.png", dpi=220)
    plt.close(fig)
    ap_df = pd.DataFrame(apr)
    ap_df.to_csv(DEST / "table_R4_auprc_by_seed.csv", index=False)
    ap_df.groupby("model", sort=False)["auprc"].agg(["mean", "std"]).to_csv(
        DEST / "table_R4_auprc_summary.csv")

    # Fixed, pre-specified thresholds: descriptive test-set sensitivity,
    # never choose the best threshold from the held-out test results.
    fixed = []
    for seed in SEEDS:
        for model in MODELS:
            _, _, y, p = records[seed, model]
            for t in THRESHOLDS:
                se, sp, fp, fn = stats(y, p, t)
                fixed.append(dict(seed=seed, model=model, threshold=t,
                                  sensitivity=se, specificity=sp, fp=fp, fn=fn,
                                  fp_per_1000_windows=1000 * fp / len(y),
                                  fn_per_1000_windows=1000 * fn / len(y)))
    fixed_df = pd.DataFrame(fixed)
    fixed_df.to_csv(DEST / "table_R4_fixed_thresholds.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, model in enumerate(MODELS):
        sub = fixed_df[fixed_df.model == model]
        avg = sub.groupby("threshold")[["fp_per_1000_windows", "fn_per_1000_windows"]].mean()
        ax.plot(avg.fp_per_1000_windows, avg.fn_per_1000_windows,
                marker=".", color=COLORS[i % len(COLORS)], label=model)
    ax.set(xlabel="False alarms per 1,000 test windows",
           ylabel="Missed positives per 1,000 test windows",
           title="Fixed-threshold decision trade-offs (mean across 3 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(DEST / "figure_R4_fp_fn_tradeoff.png", dpi=220)
    plt.close(fig)

    # Decision-region map: compare mean held-out utility at the SAME
    # pre-specified threshold and alpha across models. Descriptive only.
    # This is NOT a model/threshold selection procedure for deployment.
    winners = []
    for alpha in ALPHAS:
        for t in THRESHOLDS:
            values = {}
            for model in MODELS:
                sub = fixed_df[(fixed_df.model == model) & (fixed_df.threshold == t)]
                values[model] = float((alpha * sub.sensitivity +
                                       (1 - alpha) * sub.specificity).mean())
            best = max(MODELS, key=lambda m: values[m])
            sorted_values = sorted(values.values(), reverse=True)
            winners.append(dict(alpha=alpha, threshold=t, winning_model=best,
                                winning_utility=values[best],
                                gap_to_runner_up=sorted_values[0] - sorted_values[1]))
    win_df = pd.DataFrame(winners)
    win_df.to_csv(DEST / "table_R4_decision_regions.csv", index=False)
    matrix = np.array([[MODELS.index(win_df[
        (win_df.alpha == a) & (win_df.threshold == t)
    ].iloc[0].winning_model) for t in THRESHOLDS] for a in ALPHAS])
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([COLORS[i % len(COLORS)] for i in range(len(MODELS))])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap,
              norm=BoundaryNorm(np.arange(-.5, len(MODELS) + .5, 1), len(MODELS)))
    ax.set_xticks(range(len(THRESHOLDS)), [f"{t:.2f}" for t in THRESHOLDS], rotation=45)
    ax.set_yticks(range(len(ALPHAS)), [f"{a:.1f}" for a in ALPHAS])
    ax.set(xlabel="Fixed classification threshold", ylabel="Sensitivity weight",
           title="Descriptive decision regions: best mean held-out utility")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[i % len(COLORS)], label=m)
                       for i, m in enumerate(MODELS)],
              loc="upper center", bbox_to_anchor=(.5, -.20), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(DEST / "figure_R4_decision_regions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Hypothetical cost ratios, thresholds ALWAYS selected on validation.
    # Not real maintenance costs; unit = cost of one false positive.
    ratios = [1, 2, 5, 10, 20, 50]
    costs = []
    for ratio in ratios:
        for seed in SEEDS:
            for model in MODELS:
                yv, pv, yt, pt = records[seed, model]
                # Select by validation empirical cost, not by test performance.
                candidates = np.round(np.arange(.05, .951, .01), 2)
                val_cost = [counts(yv, pv, t)[1] + ratio * counts(yv, pv, t)[2]
                            for t in candidates]
                chosen = float(candidates[int(np.argmin(val_cost))])
                tn, fp, fn, tp = counts(yt, pt, chosen)
                costs.append(dict(seed=seed, model=model, fn_to_fp_cost_ratio=ratio,
                                  validation_selected_threshold=chosen,
                                  test_fp=fp, test_fn=fn,
                                  test_cost_per_1000_windows=1000*(fp+ratio*fn)/len(yt)))
    cost_df = pd.DataFrame(costs)
    cost_df.to_csv(DEST / "table_R4_hypothetical_costs.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, model in enumerate(MODELS):
        sub = cost_df[cost_df.model == model].groupby("fn_to_fp_cost_ratio")[
            "test_cost_per_1000_windows"].mean()
        ax.plot(sub.index, sub.values, marker="o",
                label=model, color=COLORS[i % len(COLORS)])
    ax.set_xscale("log")
    ax.set(xlabel="Hypothetical cost ratio: missed positive / false alarm",
           ylabel="Mean hypothetical cost per 1,000 test windows",
           title="Validation-selected decisions under hypothetical costs")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(DEST / "figure_R4_hypothetical_costs.png", dpi=220)
    plt.close(fig)

    (DEST / "INTERPRETATION_NOTES.txt").write_text(
        "All outputs use L30/H30 and held-out test predictions from three seeds.\n"
        "PR curves show seed 13 only; AUPRC tables report all three seeds.\n"
        "Fixed-threshold decision map uses held-out results DESCRIPTIVELY, not for selecting a model.\n"
        "Cost analysis chooses each threshold on validation, then evaluates on held-out test.\n"
        "Cost ratios are hypothetical, not empirically measured maintenance costs.\n"
        "False alarms and missed positives count WINDOW-LEVEL predictions, not unique engine events.\n"
        "Seeds share the same benchmark; they are not independent external datasets.\n"
        "No real-world validation or new fusion architecture is claimed.\n",
        encoding="utf-8"
    )
    print(f"Generated 4 figures and 5 tables in {DEST}")


if __name__ == "__main__":
    main()
