"""
CSC345 - Project Phase 2: figures for the presentation.
Reads outputs/model_results.json + outputs/oof_chosen.npy written by spaceship_model.py.

    python spaceship_model_figures.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)

# same palette / style as Phase 1 (checked for protan / deutan / tritan separation)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e6e5e1", "#ffffff"
BLUE, ORANGE, VIOLET, GRAY = "#2a78d6", "#eb6834", "#4a3aa7", "#b9b7b0"
plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 220,
})

R = json.load(open("outputs/model_results.json"))
cv, log = R["cv"], R["log"]
chosen = log["chosen_model"]
y = pd.read_csv("train.csv")["Transported"].astype(int).values
oof = np.load("outputs/oof_chosen.npy")


# ----------------------------------------------------------------- F1 model comparison (dot plot)
def fig_models():
    names = [k for k in cv if k != "Baseline (majority)"]
    names.sort(key=lambda k: cv[k]["acc_mean"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, k in enumerate(names):
        m, s = cv[k]["acc_mean"] * 100, cv[k]["acc_std"] * 100
        col = BLUE if k == chosen else INK2
        ax.errorbar(m, i, xerr=s, fmt="none", ecolor=col if k == chosen else GRAY,
                    elinewidth=2, capsize=4, zorder=2)
        ax.scatter(m, i, s=110, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(m + s + 0.15, i, f"{m:.1f}%", va="center", fontsize=13,
                fontweight="bold" if k == chosen else "normal", color=INK)
        tr = cv[k]["train_acc"] * 100
        ax.text(83.9, i, f"{tr:.0f}%", va="center", ha="right", fontsize=11.5, color=MUTED)
    ax.text(83.9, len(names) - 0.35, "train acc.", ha="right", fontsize=10.5, color=MUTED)
    ax.set_yticks(range(len(names)), names, fontsize=13, color=INK)
    ax.set_xlim(77.5, 84)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_xlabel("Validation accuracy (mean ± 1 SD over 15 grouped folds)")
    ax.grid(axis="y", visible=False)
    base = cv["Baseline (majority)"]["acc_mean"] * 100
    fig.text(0.008, 0.012,
             f"Majority-class baseline = {base:.1f}% (off the scale) · StratifiedGroupKFold, 5 folds × 3 repeats, "
             "grouped by travel group · x-axis does not start at 0 (dot plot)",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig("figures/m1_model_comparison.png"); plt.close(fig)


# ----------------------------------------------------------------- F2 permutation importance
PRETTY = {"log_Spa": "Spa spend", "log_VRDeck": "VRDeck spend", "log_RoomService": "RoomService spend",
          "log_FoodCourt": "FoodCourt spend", "log_ShoppingMall": "ShoppingMall spend",
          "log_Luxury": "Luxury spend (sum)", "log_Basic": "Basic spend (sum)",
          "log_TotalSpend": "Total spend", "CabinNum": "Cabin number", "CabinRegion": "Cabin region",
          "HomePlanet": "Home planet", "CryoSleep": "CryoSleep", "Deck": "Deck", "Side": "Side (P/S)",
          "Age": "Age", "GroupSize": "Group size", "FamilySize": "Family size", "NoSpend": "Spent nothing",
          "Destination": "Destination", "VIP": "VIP", "Solo": "Travels alone", "IsChild": "Child (≤12)"}


def fig_importance():
    pi = pd.Series(log["permutation_importance"]).sort_values(ascending=False).head(10)[::-1] * 100
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh([PRETTY.get(k, k) for k in pi.index], pi.values, color=BLUE, height=0.62)
    for i, v in enumerate(pi.values):
        ax.text(v + 0.08, i, f"{v:.1f}", va="center", fontsize=11.5, color=INK)
    ax.set_xlabel("Accuracy lost when the feature is shuffled (percentage points)")
    ax.set_xlim(0, pi.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=12.5, labelcolor=INK)
    fig.text(0.008, 0.012, f"Permutation importance · {log['grouped_vs_naive_cv']['model']} · "
             "one held-out grouped fold, 10 shuffles per feature · top 10 of 22 features",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig("figures/m2_feature_importance.png"); plt.close(fig)


# ----------------------------------------------------------------- F3 confusion matrix
def fig_confusion():
    cm = np.array(log["oof_confusion"])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    shade = np.array([[1, 0], [0, 1]])
    ax.imshow(shade, cmap=plt.matplotlib.colors.ListedColormap(["#fbe3d9", "#d6e6fa"]))
    lab = [["Correct", "False alarm"], ["Missed", "Correct"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.08, f"{cm[i, j]:,}", ha="center", va="center", fontsize=22,
                    fontweight="bold", color=INK)
            ax.text(j, i + 0.2, f"{lab[i][j]} · {cm[i, j] / cm[i].sum() * 100:.0f}% of row",
                    ha="center", va="center", fontsize=10.5, color=INK2)
    ax.set_xticks([0, 1], ["Stayed", "Transported"])
    ax.set_yticks([0, 1], ["Stayed", "Transported"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.01, 0.012, f"Out-of-fold predictions on all 8,693 train rows · {chosen} · "
             f"accuracy {log['oof_accuracy'] * 100:.1f}%", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig("figures/m3_confusion.png"); plt.close(fig)


# ----------------------------------------------------------------- F4 accuracy by subgroup (bias check)
def fig_subgroups():
    tr = pd.read_csv("train.csv")
    ok = (oof >= 0.5).astype(int) == y
    spend = tr[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
    groups = [("Earth", tr.HomePlanet == "Earth"), ("Europa", tr.HomePlanet == "Europa"),
              ("Mars", tr.HomePlanet == "Mars"), ("In CryoSleep", tr.CryoSleep == True),
              ("Awake, spent > 0", (tr.CryoSleep == False) & (spend > 0)),
              ("Awake, spent 0", (tr.CryoSleep == False) & (spend == 0))]
    labels, acc, ns = [], [], []
    for name, mask in groups:
        mask = mask.fillna(False).values
        labels.append(name); acc.append(ok[mask].mean() * 100); ns.append(mask.sum())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = [GRAY] * 3 + [BLUE, BLUE, ORANGE]
    xs = np.arange(len(labels))
    ax.bar(xs, acc, color=cols, width=0.62)
    for x, a, n in zip(xs, acc, ns):
        ax.text(x, a + 1.2, f"{a:.0f}%", ha="center", fontsize=14, fontweight="bold", color=INK)
        ax.text(x, 3, f"n={n:,}", ha="center", fontsize=10.5, color=INK if cols[x] == GRAY else SURFACE)
    overall = ok.mean() * 100
    ax.axhline(overall, color=INK2, lw=1.2)
    ax.text(len(labels) - 0.45, overall + 1.2, f"overall {overall:.1f}%", ha="right", fontsize=11, color=INK2)
    ax.axvline(2.5, color=GRID, lw=1.5, ls="--")
    ax.text(1, 112, "By home planet", ha="center", fontsize=12, color=INK2)
    ax.text(4, 112, "By CryoSleep / spending", ha="center", fontsize=12, color=INK2)
    ax.set_xticks(xs, labels, fontsize=11.5, color=INK)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_ylabel("Model accuracy")
    ax.grid(axis="x", visible=False)
    fig.text(0.008, 0.012, "Out-of-fold accuracy by subgroup · rows with a missing grouping value are left out "
             "of that bar only", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig("figures/m4_accuracy_by_subgroup.png"); plt.close(fig)
    return dict(zip(labels, [round(a, 1) for a in acc])), dict(zip(labels, [int(n) for n in ns]))


fig_models(); fig_importance(); fig_confusion()
sub_acc, sub_n = fig_subgroups()
print("subgroup accuracy:", sub_acc, sub_n)
print("figures/m1..m4 written")
