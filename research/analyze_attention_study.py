"""
Summarise research/results/attention_study.csv into a table and figures.

Outputs (research/results/):
  summary_table.csv / summary_table.md   per model × method, mean with 95% bootstrap CI
  fig_attention_split.png                 where attention goes: lesion / healthy leaf / background
  fig_camarea_vs_lesion.png               "affected area" from Grad-CAM vs measured lesion area
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODELS = ["MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121"]
COLORS = {"MobileNetV2": "#2E7D32", "ResNet50": "#2563A8", "EfficientNetB0": "#B45309", "DenseNet121": "#6D28D9"}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1])


def main(path=os.path.join(RES, "attention_study.csv")):
    df = pd.read_csv(path)
    for c in ("focus_ratio", "pointing_hit", "iou_05"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    dis = df[(df.healthy == 0) & (df.lesion_pct >= 1.0)]
    hea = df[df.healthy == 1]

    rows = []
    for m in MODELS:
        for meth in ("Grad-CAM", "Grad-CAM++"):
            d = dis[(dis.model == m) & (dis.method == meth)]
            h = hea[(hea.model == m) & (hea.method == meth)]
            if d.empty:
                continue
            r = {"Model": m, "Method": meth, "n_diseased": len(d),
                 "Accuracy (sample)": df[(df.model == m) & (df.method == meth)].correct.mean()}
            for col, label in (("e_lesion", "Attention on lesions"), ("e_healthy", "on healthy leaf"),
                               ("e_bg", "on background"), ("focus_ratio", "Lesion focus ratio"),
                               ("pointing_hit", "Pointing-game hit rate"), ("iou_05", "IoU (CAM>0.5 vs lesion)")):
                mean, lo, hi = boot_ci(d[col])
                r[label] = mean
                r[label + " CI"] = f"[{lo:.3f}, {hi:.3f}]"
            r["Background attention (healthy leaves)"] = h.e_bg.mean() if len(h) else np.nan
            r["Lesion share of leaf"] = d.lesion_pct.mean() / 100
            r["Spearman(CAM area, lesion area)"] = spearman(d.cam_area_pct, d.lesion_pct)
            r["CAM 'affected area' on healthy leaves (%)"] = h.cam_area_pct.mean() if len(h) else np.nan
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "summary_table.csv"), index=False)

    show = ["Model", "Method", "Accuracy (sample)", "Attention on lesions", "on healthy leaf", "on background",
            "Lesion share of leaf", "Lesion focus ratio", "Pointing-game hit rate", "IoU (CAM>0.5 vs lesion)",
            "Spearman(CAM area, lesion area)", "CAM 'affected area' on healthy leaves (%)"]
    with open(os.path.join(RES, "summary_table.md"), "w") as fh:
        fh.write("| " + " | ".join(show) + " |\n|" + "---|" * len(show) + "\n")
        for _, r in tab.iterrows():
            cells = []
            for c in show:
                v = r[c]
                cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            fh.write("| " + " | ".join(cells) + " |\n")
    print(tab[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ── Figures ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout="constrained")
    for ax, meth in zip(axes, ("Grad-CAM", "Grad-CAM++")):
        t = tab[tab.Method == meth]
        y = np.arange(len(t))
        les, hea_, bg = t["Attention on lesions"].values, t["on healthy leaf"].values, t["on background"].values
        ax.barh(y, les, color="#B91C1C", label="Diseased tissue")
        ax.barh(y, hea_, left=les, color="#86B98A", label="Healthy leaf")
        ax.barh(y, bg, left=les + hea_, color="#C9D2BF", label="Background")
        share = t["Lesion share of leaf"].values
        ax.scatter(share, y, marker="|", s=400, color="black", zorder=3, label="Lesion share of leaf area")
        ax.set_yticks(y, t.Model)
        ax.set_xlim(0, 1)
        ax.set_title(meth)
        ax.set_xlabel("Share of attention")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Where do the models look on diseased leaves?", fontweight="bold")
    fig.savefig(os.path.join(RES, "fig_attention_split.png"), dpi=200, bbox_inches="tight")

    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 3.6), sharex=True, sharey=True)
    d = df[(df.method == "Grad-CAM++")]
    for ax, m in zip(axes, MODELS):
        dm = d[d.model == m]
        ax.scatter(dm[dm.healthy == 0].lesion_pct, dm[dm.healthy == 0].cam_area_pct, s=8, alpha=0.5,
                   color=COLORS[m], label="Diseased")
        ax.scatter(dm[dm.healthy == 1].lesion_pct, dm[dm.healthy == 1].cam_area_pct, s=8, alpha=0.6,
                   color="black", marker="x", label="Healthy")
        ax.plot([0, 100], [0, 100], color="#999", lw=1, ls="--")
        rho = spearman(dm[dm.healthy == 0].cam_area_pct, dm[dm.healthy == 0].lesion_pct)
        ax.set_title(f"{m}\nSpearman ρ = {rho:.2f}")
        ax.set_xlabel("Measured lesion area (% of leaf)")
    axes[0].set_ylabel("Grad-CAM++ 'affected area' (% of leaf)")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_camarea_vs_lesion.png"), dpi=200, bbox_inches="tight")
    print("figures saved")


if __name__ == "__main__":
    main(*sys.argv[1:])
