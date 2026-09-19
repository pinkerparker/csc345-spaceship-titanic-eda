# CSC345 Project Phase 1 — Spaceship Titanic (EDA & Data Visualization)

Exploratory Data Analysis and data visualization of the Kaggle
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) dataset
(8,693 labelled passengers). One script reproduces every number and figure used in our
presentation.

**Team:** [Member 1] – [ID] · [Member 2] – [ID] · [Member 3] – [ID] · [Member 4] – [ID]

## The question

In 2912 the Spaceship Titanic hit a spacetime anomaly and about half of its passengers
were transported to another dimension. Who vanished, and what did they have in common?

## Two findings we present

**1. CryoSleep is the strongest single signal — on every home planet.**

| Home planet | Awake | In CryoSleep | More likely |
|---|---|---|---|
| Mars | 28% | 91% | 3.3× |
| Europa | 40% | 99% (901 of 911) | 2.5× |
| Earth | 32% | 66% | 2.0× |
| **Overall** | **33%** | **82%** | **2.5×** |

Because the gap holds inside every planet, the effect is not a home-planet artefact.

**2. Among awake passengers, *where* they spent matters more than *how much*.**

| Credits spent | Luxury (RoomService + Spa + VRDeck) | Basic (FoodCourt + ShoppingMall) |
|---|---|---|
| under 250 | 64% transported | 13% transported |
| over 4,000 | **2% transported** | **69% transported** |

The two directions are opposite, so collapsing everything into one "total spend" feature
would cancel the signal out.

## Data problems we found and how we handled them

| # | Problem | Evidence | Handling |
|---|---|---|---|
| 1 | Missing values in every column | ~2% per column, but 24% of rows affected | No row dropping; rule-based imputation; every figure reports its `n` |
| 2 | Compound columns | `Cabin` = deck/num/side, `PassengerId` = group_member | Split into `Deck`, `CabinNum`, `Side`, `Group`, `GroupSize` |
| 3 | Extreme skew in spending | 61–64% zeros, skewness 6–13, max 29,813 credits | Spend bins / log scale instead of means |
| 4 | Hidden logical rules | 0 CryoSleep passengers and 0 children ≤ 12 ever spent | 347 sleepers with missing spend → 0; 119 spenders with missing CryoSleep → awake |
| 5 | Rare categories & confounding | VIP 2.3% (no Earth VIPs); deck T = 5 people; Earth only on decks E–G | Report group sizes and 95% CIs; compare within groups |
| 6 | Types & odd values | Booleans with NaN load as object; 178 passengers aged 0; 20 repeated names | Cast types; keep age 0 (consistent with zero spend); `Name` unused |

## How to run

```bash
# 1. get the data from Kaggle (login required) and unzip it next to the script
#    https://www.kaggle.com/competitions/spaceship-titanic/data
#    you need train.csv and test.csv

# 2. install the libraries
pip install -r requirements.txt

# 3. run
python spaceship_eda.py
```

Output:

```
figures/plot1_cryosleep_homeplanet.png    presented — visualization 1
figures/plot2_luxury_vs_basic_spend.png   presented — visualization 2
figures/a1_missing.png                    appendix — missing values per column
figures/a2_spend_dist.png                 appendix — spending distribution (log)
figures/a3_age.png                        appendix — age by outcome
figures/a4_deck_side.png                  appendix — deck × side heatmap
eda_stats.json                            every number quoted on the slides
```

The dataset itself is **not** included in this repository (Kaggle competition rules);
`.gitignore` keeps the CSV files out.

## Visualization design choices

* Bars start at zero, and every bar or bin carries its sample size.
* 95% **Wilson** confidence intervals — accurate near 0% and 100%, unlike the normal interval.
* One y-axis only (no dual-axis charts).
* Spend bins are equal-width and every bin holds at least 300 passengers.
* Plot 2 uses awake passengers only, since sleepers cannot spend — including them would
  bias the zero-spend group.
* Colours were checked for protan / deutan / tritan separation.

## Next: Phase 2 (modeling)

Rule-based imputation · features `Deck`, `Side`, `GroupSize`, luxury vs basic spend ·
log-transformed spending · `CryoSleep × HomePlanet` interaction · logistic regression,
random forest, gradient boosting.

## References

* Dataset — Kaggle, *Spaceship Titanic* (2022): <https://www.kaggle.com/competitions/spaceship-titanic>
* Tools — Python 3, [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org), [Matplotlib](https://matplotlib.org)
* Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
* Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212.
* Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. <https://clauswilke.com/dataviz>
* Okabe & Ito, colour-blind-safe palette guidance: <https://jfly.uni-koeln.de/color/>
