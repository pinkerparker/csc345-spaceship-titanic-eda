"""
CSC345 - Project Phase 2: Modeling
Dataset: Kaggle "Spaceship Titanic"  (https://www.kaggle.com/competitions/spaceship-titanic)
Task   : binary classification of `Transported`; Kaggle metric = classification accuracy

Pipeline
--------
1. Load train.csv + test.csv
2. Rule-based cleaning that uses NO target information (findings from Phase 1)
      - CryoSleep / age <= 12 -> spend = 0 ;  spent > 0 -> CryoSleep = False
      - HomePlanet, cabin Side, Deck filled from travel group / surname (100% consistent)
3. Feature engineering  (Deck, Side, CabinNum, GroupSize, luxury vs basic spend, log spend ...)
4. Remaining gaps -> median / most-frequent imputation INSIDE the sklearn Pipeline,
   so it is fitted only on the training folds (no leakage)
5. Validation = StratifiedGroupKFold (5 folds x 3 repeats) grouped by travel group,
   because train and test share no groups at all
6. Compare models: majority baseline, logistic regression, random forest,
   HistGradientBoosting, LightGBM, XGBoost, CatBoost + soft-voting ensemble
7. Refit the chosen model on all training rows -> submission.csv

How to run
----------
    pip install -r requirements.txt
    python spaceship_model.py     # train.csv / test.csv in the same folder
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
os.makedirs("outputs", exist_ok=True)

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LUXURY = ["RoomService", "Spa", "VRDeck"]
BASIC = ["FoodCourt", "ShoppingMall"]

# =========================================================================== 1. load
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["Transported"].astype(int).values
n_train = len(train)

# features are built on train+test together. Only label-free information is shared
# (group / surname / cabin), never the target.
full = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
log = {"missing_before": int(full[["HomePlanet", "CryoSleep", "Cabin", "Destination", "Age",
                                   "VIP"] + SPEND].isna().sum().sum())}

# =========================================================================== 2. parse + rule-based cleaning
full["Group"] = full["PassengerId"].str[:4]
full["GroupSize"] = full.groupby("Group")["Group"].transform("size")
full[["Deck", "CabinNum", "Side"]] = full["Cabin"].str.split("/", expand=True)
full["CabinNum"] = pd.to_numeric(full["CabinNum"])
full["Surname"] = full["Name"].str.split().str[-1]
for c in ["CryoSleep", "VIP"]:
    full[c] = full[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

fills = {}

def fill(col, mask, values, rule):
    before = full[col].isna().sum()
    full.loc[mask, col] = values[mask] if isinstance(values, pd.Series) else values
    fills[rule] = int(before - full[col].isna().sum())

# rule A: anyone in CryoSleep spends nothing
for c in SPEND:
    fill(c, full[c].isna() & (full.CryoSleep == 1), 0.0, f"spend<-0 (CryoSleep) {c}")
# rule B: children aged <= 12 never spend
for c in SPEND:
    fill(c, full[c].isna() & (full.Age <= 12), 0.0, f"spend<-0 (age<=12) {c}")
# rule C: spent something -> was awake
spent = full[SPEND].sum(axis=1) > 0
fill("CryoSleep", full.CryoSleep.isna() & spent, 0.0, "CryoSleep<-0 (spent)")

# rule D: HomePlanet / Side are identical within a travel group, HomePlanet within a surname
for col, key in [("HomePlanet", "Group"), ("HomePlanet", "Surname"), ("Side", "Group"),
                 ("Deck", "Group")]:
    mode = full.groupby(key)[col].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    fill(col, full[col].isna() & full[key].notna(), full[key].map(mode), f"{col}<-{key}")
# rule E: Earth passengers only live on decks E-G (Phase 1)  -> deck G is the mode
fill("Deck", full.Deck.isna() & (full.HomePlanet == "Earth"), "G", "Deck<-G (Earth)")
# rule F: no Earth VIPs
fill("VIP", full.VIP.isna() & (full.HomePlanet == "Earth"), 0.0, "VIP<-0 (Earth)")

log["rule_fills"] = {k: v for k, v in fills.items() if v > 0}
log["rule_fills_total"] = sum(log["rule_fills"].values())

# =========================================================================== 3. feature engineering
full["Luxury"] = full[LUXURY].sum(axis=1, min_count=1)
full["Basic"] = full[BASIC].sum(axis=1, min_count=1)
full["TotalSpend"] = full[SPEND].sum(axis=1, min_count=1)
for c in SPEND + ["Luxury", "Basic", "TotalSpend"]:
    full[f"log_{c}"] = np.log1p(full[c])
full["NoSpend"] = (full["TotalSpend"].fillna(0) == 0).astype(float)
full["Solo"] = (full["GroupSize"] == 1).astype(float)
full["FamilySize"] = full.groupby("Surname")["Surname"].transform("size")
full["CabinRegion"] = (full["CabinNum"] // 300)          # cabin blocks along the ship
full["IsChild"] = (full["Age"] <= 12).astype(float).where(full.Age.notna())

NUM = ["Age", "CabinNum", "CabinRegion", "GroupSize", "FamilySize",
       "log_RoomService", "log_FoodCourt", "log_ShoppingMall", "log_Spa", "log_VRDeck",
       "log_Luxury", "log_Basic", "log_TotalSpend"]
BIN = ["CryoSleep", "VIP", "NoSpend", "Solo", "IsChild"]
CAT = ["HomePlanet", "Destination", "Deck", "Side"]
FEATURES = NUM + BIN + CAT
log["features"] = {"numeric": NUM, "binary": BIN, "categorical": CAT, "n_total": len(FEATURES)}
# Not used: PassengerId / Name (identifiers), Cabin (split), Group (identifier, used for CV only)

X_all = full[FEATURES].copy()
for c in CAT:
    X_all[c] = X_all[c].astype("object")
X, X_test = X_all.iloc[:n_train].reset_index(drop=True), X_all.iloc[n_train:].reset_index(drop=True)
groups = full["Group"].iloc[:n_train].values
log["missing_after_rules"] = int(X.isna().sum().sum() + X_test.isna().sum().sum())


# =========================================================================== 4. preprocessing (fit per fold)
def preprocessor(scale):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ], verbose_feature_names_out=False)


def make(model, scale=False):
    return Pipeline([("prep", preprocessor(scale)), ("model", model)])


MODELS = {
    "Baseline (majority)": make(DummyClassifier(strategy="most_frequent")),
    "Logistic Regression": make(LogisticRegression(C=0.5, max_iter=2000), scale=True),
    "Random Forest": make(RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                 max_features="sqrt", n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": make(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400,
                                                                max_leaf_nodes=24, l2_regularization=1.0,
                                                                random_state=SEED)),
    "LightGBM": make(LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=24,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                    reg_lambda=1.0, random_state=SEED, verbose=-1)),
    "XGBoost": make(XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                  random_state=SEED, n_jobs=-1, eval_metric="logloss")),
    "CatBoost": make(CatBoostClassifier(iterations=800, learning_rate=0.04, depth=6,
                                        random_seed=SEED, verbose=0, thread_count=-1)),
}


# =========================================================================== 5. validation
N_SPLITS, N_REPEATS = 5, 3


def cv_splits(seed, grouped=True):
    if grouped:
        return list(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y, groups))
    return list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, y))


SPLITS = [s for r in range(N_REPEATS) for s in cv_splits(SEED + r)]


def evaluate(pipe, splits=SPLITS, keep_oof=False):
    accs, aucs, train_accs = [], [], []
    oof = np.zeros(len(y))
    for tr, va in splits:
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        accs.append(accuracy_score(y[va], p >= 0.5))
        aucs.append(roc_auc_score(y[va], p) if len(np.unique(p)) > 1 else 0.5)
        train_accs.append(accuracy_score(y[tr], m.predict(X.iloc[tr])))
        if keep_oof:
            oof[va] += p / N_REPEATS
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)), "train_acc": float(np.mean(train_accs)),
            "fold_accs": [float(a) for a in accs]}, oof


results, oofs = {}, {}
for name, pipe in MODELS.items():
    t = time.time()
    results[name], oofs[name] = evaluate(pipe, keep_oof=True)
    results[name]["seconds"] = round(time.time() - t, 1)
    r = results[name]
    print(f"{name:22s} acc {r['acc_mean']:.4f} ± {r['acc_std']:.4f}   auc {r['auc_mean']:.4f}   "
          f"train {r['train_acc']:.4f}   ({r['seconds']}s)")

# --- small random search for the gradient-boosting models.
# Tuned on the folds of repeat #1 only; the score we report uses all 15 folds,
# and repeats #2-3 use different splits, which limits optimistic selection bias.
rng = np.random.default_rng(SEED)
TUNE_SPACE = {
    "XGBoost": lambda: XGBClassifier(n_estimators=int(rng.integers(300, 1200)),
                                     learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05])),
                                     max_depth=int(rng.integers(3, 8)),
                                     min_child_weight=float(rng.choice([1, 3, 5, 10])),
                                     subsample=float(rng.uniform(0.6, 1.0)),
                                     colsample_bytree=float(rng.uniform(0.4, 0.9)),
                                     reg_lambda=float(rng.choice([0.5, 1, 3, 10])),
                                     random_state=SEED, n_jobs=-1, eval_metric="logloss"),
    "LightGBM": lambda: LGBMClassifier(n_estimators=int(rng.integers(300, 1200)),
                                       learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05])),
                                       num_leaves=int(rng.integers(8, 48)),
                                       min_child_samples=int(rng.choice([10, 20, 40, 80])),
                                       subsample=float(rng.uniform(0.6, 1.0)), subsample_freq=1,
                                       colsample_bytree=float(rng.uniform(0.4, 0.9)),
                                       reg_lambda=float(rng.choice([0.5, 1, 3, 10])),
                                       random_state=SEED, verbose=-1),
}
N_TRIALS = 20
tuning = {}
for name, sampler in TUNE_SPACE.items():
    best_acc, best_model = -1, None
    for _ in range(N_TRIALS):
        cand = make(sampler())
        a, _ = evaluate(cand, splits=SPLITS[:N_SPLITS])
        if a["acc_mean"] > best_acc:
            best_acc, best_model = a["acc_mean"], cand
    tuned_name = f"{name} (tuned)"
    MODELS[tuned_name] = best_model
    results[tuned_name], oofs[tuned_name] = evaluate(best_model, keep_oof=True)
    tuning[tuned_name] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                          for k, v in best_model.named_steps["model"].get_params().items()
                          if k in ("n_estimators", "learning_rate", "max_depth", "num_leaves",
                                   "min_child_weight", "min_child_samples", "subsample",
                                   "colsample_bytree", "reg_lambda")}
    r = results[tuned_name]
    print(f"{tuned_name:22s} acc {r['acc_mean']:.4f} ± {r['acc_std']:.4f}   auc {r['auc_mean']:.4f}   "
          f"train {r['train_acc']:.4f}   (search acc {best_acc:.4f})")
log["tuning"] = {"trials_per_model": N_TRIALS, "best_params": tuning}

# soft-voting ensemble of the three strongest models
top3 = sorted([k for k in MODELS if k not in ("Baseline (majority)",)],
              key=lambda k: -results[k]["acc_mean"])[:3]
ensemble = make(VotingClassifier([(k, clone(MODELS[k].named_steps["model"])) for k in top3],
                                 voting="soft"))
results["Ensemble (top-3 soft vote)"], oofs["Ensemble (top-3 soft vote)"] = evaluate(ensemble, keep_oof=True)
results["Ensemble (top-3 soft vote)"]["members"] = top3
r = results["Ensemble (top-3 soft vote)"]
print(f"{'Ensemble ' + str(top3):22s} acc {r['acc_mean']:.4f} ± {r['acc_std']:.4f}   auc {r['auc_mean']:.4f}")

# --- bias check: how optimistic is an ordinary (non-grouped) K-fold?
leak_splits = [s for r in range(N_REPEATS) for s in cv_splits(SEED + r, grouped=False)]
best_single = max([k for k in MODELS if k != "Baseline (majority)"], key=lambda k: results[k]["acc_mean"])
naive, _ = evaluate(MODELS[best_single], splits=leak_splits)
log["grouped_vs_naive_cv"] = {"model": best_single,
                             "grouped_acc": results[best_single]["acc_mean"],
                             "naive_acc": naive["acc_mean"]}
print(f"naive KFold for {best_single}: {naive['acc_mean']:.4f} (grouped {results[best_single]['acc_mean']:.4f})")

# --- feature-group ablation on the best single model (does Phase-1 insight help?)
ablation = {}
def drop_feats(pipe, drop):
    keep_num = [c for c in NUM if c not in drop]
    keep_bin = [c for c in BIN if c not in drop]
    keep_cat = [c for c in CAT if c not in drop]
    p = clone(pipe)
    p.named_steps["prep"].transformers = [
        ("num", p.named_steps["prep"].transformers[0][1], keep_num),
        ("bin", p.named_steps["prep"].transformers[1][1], keep_bin),
        ("cat", p.named_steps["prep"].transformers[2][1], keep_cat)]
    return p

ABLATIONS = {
    "- Luxury/Basic split": ["log_Luxury", "log_Basic"],
    "- cabin features (Deck/Side/Num)": ["Deck", "Side", "CabinNum", "CabinRegion"],
    "- group/family features": ["GroupSize", "FamilySize", "Solo"],
    "- CryoSleep": ["CryoSleep"],
}
for label, cols in ABLATIONS.items():
    a, _ = evaluate(drop_feats(MODELS[best_single], cols), splits=SPLITS[:N_SPLITS])
    ablation[label] = a["acc_mean"]
ref, _ = evaluate(MODELS[best_single], splits=SPLITS[:N_SPLITS])
ablation["all features"] = ref["acc_mean"]
log["ablation"] = {"model": best_single, **ablation}
print("ablation:", {k: round(v, 4) for k, v in ablation.items()})

# =========================================================================== 6. choose + refit + predict
candidates = {k: v for k, v in results.items() if k != "Baseline (majority)"}
chosen = max(candidates, key=lambda k: candidates[k]["acc_mean"])
final = ensemble if chosen.startswith("Ensemble") else MODELS[chosen]
final = clone(final).fit(X, y)
pred = final.predict(X_test).astype(bool)
pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred}).to_csv(
    "outputs/submission.csv", index=False)

oof_best = oofs[chosen]
cm = confusion_matrix(y, oof_best >= 0.5)
log["chosen_model"] = chosen
log["oof_confusion"] = cm.tolist()
log["oof_accuracy"] = float(accuracy_score(y, oof_best >= 0.5))
log["test_pred_transported_rate"] = float(pred.mean())

# --- error analysis: accuracy by subgroup (fairness / where the model struggles)
ea = pd.DataFrame({"y": y, "p": (oof_best >= 0.5).astype(int),
                   "HomePlanet": full.HomePlanet.iloc[:n_train].values,
                   "CryoSleep": full.CryoSleep.iloc[:n_train].values,
                   "NoSpend": full.NoSpend.iloc[:n_train].values})
ea["ok"] = ea.y == ea.p
log["acc_by_planet"] = ea.groupby("HomePlanet").ok.mean().round(4).to_dict()
log["acc_by_cryo"] = ea.groupby("CryoSleep").ok.mean().round(4).rename(index=str).to_dict()
log["acc_awake_nospend"] = float(ea[(ea.CryoSleep == 0) & (ea.NoSpend == 1)].ok.mean())
log["n_awake_nospend"] = int(((ea.CryoSleep == 0) & (ea.NoSpend == 1)).sum())

# --- permutation importance of the best single model on a held-out grouped fold
tr, va = SPLITS[0]
m = clone(MODELS[best_single]).fit(X.iloc[tr], y[tr])
pi = permutation_importance(m, X.iloc[va], y[va], n_repeats=10, random_state=SEED,
                            scoring="accuracy", n_jobs=-1)
log["permutation_importance"] = dict(sorted(
    {f: float(v) for f, v in zip(FEATURES, pi.importances_mean)}.items(), key=lambda kv: -kv[1]))

with open("outputs/model_results.json", "w") as f:
    json.dump({"cv": results, "log": log}, f, indent=2, default=str)
np.save("outputs/oof_chosen.npy", oof_best)

print("\nChosen:", chosen, f"CV acc {results[chosen]['acc_mean']:.4f}")
print("submission.csv rows:", len(pred), " transported rate:", round(pred.mean(), 3))
