"""
CSC345 / BIF633 - Project Phase 1: EDA & Data Visualization
Dataset: Kaggle "Spaceship Titanic"  (https://www.kaggle.com/competitions/spaceship-titanic)

What this script does
---------------------
1. Loads train.csv / test.csv
2. Profiles the data (missing values, skew, hidden rules) -> eda_stats.json
3. Builds the two visualizations we present:
      figures/plot1_cryosleep_homeplanet.png   CryoSleep x HomePlanet
      figures/plot2_luxury_vs_basic_spend.png  Luxury vs basic spending (awake passengers)
4. Builds four supporting figures used in the appendix (a1..a4)

How to run
----------
    pip install -r requirements.txt
    python spaceship_eda.py          # train.csv and test.csv must be in the same folder

Team: [Member 1], [Member 2], [Member 3], [Member 4]
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- setup
os.makedirs("figures", exist_ok=True)

# colour palette (checked for protan / deutan / tritan separation)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e6e5e1", "#ffffff"
BLUE, ORANGE, VIOLET, GRAY = "#2a78d6", "#eb6834", "#4a3aa7", "#b9b7b0"
CHIP_BG = "#eaf2fd"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 220,
})

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LUXURY = ["RoomService", "Spa", "VRDeck"]          # private / in-cabin services
BASIC = ["FoodCourt", "ShoppingMall"]              # public areas of the ship
PLANETS = ["Earth", "Europa", "Mars"]


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion (k successes out of n).

    Wilson is used instead of the textbook normal interval because our rates
    go very close to 0% and 100%, where the normal interval breaks down.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


# --------------------------------------------------------------------------- 1. load
df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
RAW_COLUMNS = list(df.columns)

# --------------------------------------------------------------------------- 2. feature engineering
# Cabin looks like "B/0/P" -> deck / cabin number / side (P = Port, S = Starboard)
df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)
df["CabinNum"] = pd.to_numeric(df["CabinNum"])

# PassengerId looks like "0001_01" -> the first 4 digits are the travel group
df["Group"] = df["PassengerId"].str[:4]
df["GroupSize"] = df.groupby("Group")["Group"].transform("size")

df["TotalSpend"] = df[SPEND].sum(axis=1, min_count=1)

# --------------------------------------------------------------------------- 3. profiling
stats = {}
stats["shape_train"] = list(df.shape)
stats["shape_test"] = list(test.shape)
stats["target_rate"] = round(df["Transported"].mean() * 100, 1)
stats["missing_pct"] = (df[RAW_COLUMNS].isna().mean() * 100).round(2).to_dict()
stats["rows_any_missing_pct"] = round(df[RAW_COLUMNS].isna().any(axis=1).mean() * 100, 1)
stats["zero_spend_pct"] = {c: round((df[c] == 0).mean() * 100, 1) for c in SPEND}
stats["skew"] = {c: round(df[c].skew(), 1) for c in SPEND}
stats["max_spend"] = {c: float(df[c].max()) for c in SPEND}

# data problems / hidden rules
stats["cryo_passengers_who_spent"] = int(((df.CryoSleep == True) & (df[SPEND].sum(axis=1) > 0)).sum())
stats["children_le12_who_spent"] = int(((df.Age <= 12) & (df[SPEND].sum(axis=1) > 0)).sum())
stats["spend_missing_but_in_cryo"] = int((df[SPEND].isna().any(axis=1) & (df.CryoSleep == True)).sum())
stats["cryo_missing_but_spent"] = int((df.CryoSleep.isna() & (df[SPEND].sum(axis=1) > 0)).sum())
stats["age_zero"] = int((df.Age == 0).sum())
stats["duplicate_names"] = int(df.Name.dropna().duplicated().sum())
stats["vip_pct"] = round((df.VIP == True).mean() * 100, 1)
stats["earth_vips"] = int(((df.HomePlanet == "Earth") & (df.VIP == True)).sum())
stats["deck_T_passengers"] = int((df.Deck == "T").sum())
stats["deck_by_planet"] = pd.crosstab(df.HomePlanet, df.Deck).to_dict()
stats["solo_travellers_pct"] = round((df.GroupSize == 1).mean() * 100, 1)


# --------------------------------------------------------------------------- 4. PLOT 1
# CryoSleep x HomePlanet -> share transported
def build_plot1():
    data = df.dropna(subset=["CryoSleep", "HomePlanet"])
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    bar_w, gap = 0.34, 0.04
    rates, out = {}, {}

    for i, (flag, label, colour) in enumerate([(False, "Awake", GRAY), (True, "In CryoSleep", BLUE)]):
        for j, planet in enumerate(PLANETS):
            group = data[(data.HomePlanet == planet) & (data.CryoSleep == flag)]["Transported"]
            k, n = int(group.sum()), len(group)
            rate = k / n * 100
            lo, hi = wilson(k, n)
            x = j + (i - 0.5) * (bar_w + gap)

            ax.bar(x, rate, bar_w, color=colour, zorder=2, label=label if j == 0 else None)
            ax.errorbar(x, rate, yerr=[[rate - lo * 100], [hi * 100 - rate]], fmt="none",
                        ecolor=INK2, elinewidth=1.2, capsize=4, zorder=3)
            ax.text(x, hi * 100 + 1.5, f"{rate:.0f}%", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", color=INK)
            ax.text(x, 2.5, f"n={n:,}", ha="center", va="bottom", fontsize=10.5,
                    color=SURFACE if flag else INK)

            rates[(planet, flag)] = rate
            out[f"{planet}_{label}"] = {"rate": round(rate, 1), "n": n,
                                        "ci": [round(lo * 100, 1), round(hi * 100, 1)]}

    # "x N more likely" chip above each planet pair
    for j, planet in enumerate(PLANETS):
        mult = rates[(planet, True)] / rates[(planet, False)]
        out[f"{planet}_multiplier"] = round(mult, 1)
        ax.text(j, 120, f"{mult:.1f}× more likely", ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=BLUE,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=CHIP_BG, edgecolor="none"))

    # callout on the strongest bar
    europa_cryo = data[(data.HomePlanet == "Europa") & (data.CryoSleep == True)]["Transported"]
    ax.annotate(f"{int(europa_cryo.sum())} of {len(europa_cryo)} Europans\nin CryoSleep vanished",
                xy=(1 + (bar_w + gap) / 2, 99), xytext=(1.55, 72), fontsize=11.5, color=INK,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1, connectionstyle="arc3,rad=-0.25"))

    overall = df.Transported.mean() * 100
    ax.axhline(overall, color=INK2, lw=1.2, zorder=1)
    ax.text(-0.62, overall + 1.5, f"all passengers {overall:.1f}%", ha="left", va="bottom",
            fontsize=11, color=INK2)

    ax.set_xticks(range(3), PLANETS, fontsize=16, color=INK)
    ax.set_xlim(-0.65, 2.7)
    ax.set_ylim(0, 142)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_ylabel("Share of passengers transported")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=13, ncol=2, bbox_to_anchor=(-0.02, 1.0))

    fig.text(0.008, 0.012,
             "Error bars = 95% Wilson confidence interval · excludes "
             f"{len(df) - len(data)} rows with a missing CryoSleep or HomePlanet value "
             f"(n = {len(data):,}) · Source: Kaggle Spaceship Titanic train.csv",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig("figures/plot1_cryosleep_homeplanet.png")
    plt.close(fig)

    out["n_used"] = len(data)
    out["overall_cryo"] = round(data[data.CryoSleep == True].Transported.mean() * 100, 1)
    out["overall_awake"] = round(data[data.CryoSleep == False].Transported.mean() * 100, 1)
    return out


# --------------------------------------------------------------------------- 5. PLOT 2
# Awake passengers only: luxury vs basic spending -> share transported
def build_plot2():
    awake = df[df.CryoSleep == False].copy()
    awake["Luxury"] = awake[LUXURY].sum(axis=1, min_count=len(LUXURY))
    awake["Basic"] = awake[BASIC].sum(axis=1, min_count=len(BASIC))

    bins = [-1, 0, 250, 500, 1000, 2000, 4000, 1e9]
    labels = ["0", "1–250", "251–500", "501–1k", "1k–2k", "2k–4k", ">4k"]

    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    xs = np.arange(len(labels))
    out = {}

    for column, colour in [("Luxury", ORANGE), ("Basic", VIOLET)]:
        sub = awake.dropna(subset=[column])
        grouped = sub.groupby(pd.cut(sub[column], bins, labels=labels), observed=False)["Transported"]
        k, n = grouped.sum().values, grouped.size().values
        rate = k / n * 100
        ci = np.array([wilson(a, b) for a, b in zip(k, n)]) * 100

        ax.fill_between(xs, ci[:, 0], ci[:, 1], color=colour, alpha=0.15, lw=0, zorder=1)
        ax.plot(xs, rate, color=colour, lw=2.5, zorder=3)
        ax.scatter(xs, rate, s=70, color=colour, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{rate[-1]:.0f}%", (xs[-1], rate[-1]), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=15, fontweight="bold", color=INK)
        ax.annotate(f"{rate[0]:.0f}%", (xs[0], rate[0]), xytext=(-12, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=13, color=INK2)

        out[column] = {"bins": labels, "rate": [round(r, 1) for r in rate],
                       "n": [int(v) for v in n], "n_total": int(len(sub))}

    ax.text(0.0, 71, "Luxury: RoomService + Spa + VRDeck", color=ORANGE,
            fontsize=13.5, fontweight="bold")
    ax.text(0.55, 3.0, "Basic: FoodCourt + ShoppingMall", color=VIOLET,
            fontsize=13.5, fontweight="bold")

    awake_rate = awake.Transported.mean() * 100
    ax.axhline(awake_rate, color=INK2, lw=1.2, zorder=0)
    ax.text(len(labels) - 0.35, awake_rate + 1.2, f"all awake passengers {awake_rate:.0f}%",
            ha="right", fontsize=11, color=INK2)

    ax.set_xticks(xs, labels, fontsize=13, color=INK)
    ax.set_xlim(-0.8, len(labels) - 0.3)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_xlabel("Credits spent on board (per passenger)")
    ax.set_ylabel("Share of passengers transported")
    ax.grid(axis="x", visible=False)

    fig.text(0.008, 0.012,
             "Awake passengers only (CryoSleep = False) · shaded band = 95% Wilson CI · every bin "
             f"has n ≥ 300 · Luxury n = {out['Luxury']['n_total']:,}, Basic n = {out['Basic']['n_total']:,}",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig("figures/plot2_luxury_vs_basic_spend.png")
    plt.close(fig)

    out["awake_rate"] = round(awake_rate, 1)
    out["awake_n"] = int(len(awake))
    return out


# --------------------------------------------------------------------------- 6. appendix figures
def build_appendix():
    # A1 - missing values per column
    miss = (df[RAW_COLUMNS].isna().mean() * 100).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(miss.index, miss.values, color=BLUE, height=0.65)
    for i, v in enumerate(miss.values):
        ax.text(v + 0.03, i, f"{v:.1f}%", va="center", fontsize=11, color=INK)
    ax.set_xlabel("% of rows missing")
    ax.set_xlim(0, 3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig("figures/a1_missing.png"); plt.close(fig)

    # A2 - spending is heavily right-skewed (log scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    total = df["TotalSpend"].dropna()
    ax.hist(np.log10(total[total > 0]), bins=40, color=BLUE, edgecolor=SURFACE)
    ax.set_xlabel("Total spend (log10 credits, spenders only)")
    ax.set_ylabel("Passengers")
    ax.set_xticks([1, 2, 3, 4], ["10", "100", "1,000", "10,000"])
    ax.text(0.02, 0.95, f"{(total == 0).mean() * 100:.0f}% of passengers spent 0",
            transform=ax.transAxes, fontsize=12, color=INK, va="top")
    fig.tight_layout(); fig.savefig("figures/a2_spend_dist.png"); plt.close(fig)

    # A3 - age distribution by outcome
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.arange(0, 82, 4)
    ax.hist(df[df.Transported].Age.dropna(), bins=edges, histtype="step", lw=2.2,
            color=BLUE, label="Transported")
    ax.hist(df[~df.Transported].Age.dropna(), bins=edges, histtype="step", lw=2.2,
            color=ORANGE, label="Not transported")
    ax.set_xlabel("Age"); ax.set_ylabel("Passengers"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/a3_age.png"); plt.close(fig)

    # A4 - deck x side heatmap (deck T dropped: only 5 passengers)
    ds = df[df.Deck != "T"].dropna(subset=["Deck", "Side"])
    pivot = pd.crosstab(ds.Deck, ds.Side, values=ds.Transported, aggfunc="mean") * 100
    counts = pd.crosstab(ds.Deck, ds.Side)
    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", BLUE, "#0d366b"])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(pivot.values, cmap=cmap, vmin=30, vmax=80, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.0f}%\nn={counts.values[i, j]}", ha="center", va="center",
                    fontsize=11, color=SURFACE if v > 55 else INK)
    ax.set_xticks([0, 1], ["Port (P)", "Starboard (S)"])
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_ylabel("Deck"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="% transported")
    fig.tight_layout(); fig.savefig("figures/a4_deck_side.png"); plt.close(fig)


# --------------------------------------------------------------------------- 7. run everything
stats["plot1"] = build_plot1()
stats["plot2"] = build_plot2()
build_appendix()

# extra breakdowns quoted on the slides
stats["rate_by_cryosleep"] = (df.groupby("CryoSleep").Transported.mean() * 100).round(1).to_dict()
stats["rate_by_planet"] = (df.groupby("HomePlanet").Transported.mean() * 100).round(1).to_dict()
stats["rate_by_destination"] = (df.groupby("Destination").Transported.mean() * 100).round(1).to_dict()
stats["rate_by_deck"] = (df.groupby("Deck").Transported.mean() * 100).round(1).to_dict()
stats["rate_by_side"] = (df.groupby("Side").Transported.mean() * 100).round(1).to_dict()
stats["rate_by_vip"] = (df.groupby("VIP").Transported.mean() * 100).round(1).to_dict()
stats["rate_by_group_size"] = (df.groupby("GroupSize").Transported.mean() * 100).round(1).to_dict()
age_bins = pd.cut(df.Age, [-1, 4, 12, 17, 25, 50, 80])
stats["rate_by_age"] = (df.groupby(age_bins, observed=True).Transported.mean() * 100).round(1).rename(index=str).to_dict()

with open("eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)

print("Done.")
print(f"  passengers          : {stats['shape_train'][0]:,}")
print(f"  transported         : {stats['target_rate']}%")
print(f"  rows with a gap     : {stats['rows_any_missing_pct']}%")
print(f"  CryoSleep vs awake  : {stats['plot1']['overall_cryo']}% vs {stats['plot1']['overall_awake']}%")
print("  figures/            : 2 main plots + 4 appendix plots")
print("  eda_stats.json      : every number used on the slides")
