"""Shared helpers: preprocessing pipelines, the three model classes, and cluster-bootstrap metrics."""

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

SEED = 20261024


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def dense_preprocessor(features, log_features=()):
    """Median imputation (+ missing indicators), signed-log for heavy-tailed variables, standardisation.
    Fitted inside each training fold only, so no test information reaches the scaler or imputer."""
    log_features = [f for f in features if f in set(log_features)]
    plain = [f for f in features if f not in set(log_features)]
    steps = []
    if log_features:
        steps.append(("log", Pipeline([
            ("slog", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), log_features))
    if plain:
        steps.append(("plain", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), plain))
    return ColumnTransformer(steps)


def model_specs(task, features, log_features=(), pos_weight=1.0):
    """Three model classes: penalised linear, gradient-boosted trees, feed-forward neural network.
    Returns {name: (estimator, param_distributions)} for randomized search."""
    if task == "classification":
        linear = Pipeline([
            ("prep", dense_preprocessor(features, log_features)),
            ("model", LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED)),
        ])
        linear_grid = {"model__C": loguniform(1e-3, 1e1), "model__penalty": ["l1", "l2"]}
        trees = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=8)
        mlp = Pipeline([
            ("prep", dense_preprocessor(features, log_features)),
            ("model", MLPClassifier(early_stopping=True, max_iter=300, random_state=SEED)),
        ])
    else:
        linear = Pipeline([
            ("prep", dense_preprocessor(features, log_features)),
            ("model", ElasticNet(max_iter=20000, random_state=SEED)),
        ])
        linear_grid = {"model__alpha": loguniform(1e-4, 1e0), "model__l1_ratio": uniform(0, 1)}
        trees = XGBRegressor(tree_method="hist", objective="reg:squarederror", random_state=SEED, n_jobs=8)
        # Standardize the target too, so the network's output scale does not slow convergence
        mlp = TransformedTargetRegressor(regressor=Pipeline([
            ("prep", dense_preprocessor(features, log_features)),
            ("model", MLPRegressor(early_stopping=True, max_iter=200, random_state=SEED)),
        ]), transformer=StandardScaler())
    tree_grid = {
        "n_estimators": randint(100, 800),
        "learning_rate": loguniform(0.01, 0.2),
        "max_depth": randint(2, 7),
        "min_child_weight": loguniform(1, 50),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(0.1, 20),
    }
    mlp_grid = {
        "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (32, 16)],
        "model__alpha": loguniform(1e-5, 1e-1),
        "model__learning_rate_init": loguniform(1e-4, 1e-2),
        "model__batch_size": [256, 512, 1024],
    }
    if task != "classification":
        mlp_grid = {f"regressor__{k}": v for k, v in mlp_grid.items()}
    return {
        "Penalized linear": (linear, linear_grid),
        "Gradient-boosted trees": (trees, tree_grid),
        "Neural network (MLP)": (mlp, mlp_grid),
    }


def forward_group_cv(df, time_col, group_col, val_blocks, gap):
    """Forward-chaining folds that are also disjoint in units.
    For fold k: validation = rows of group-fold k whose time falls in val_blocks[k] = (start, end);
    training = rows of the other group-folds with time <= start - gap (labels observable before validation).
    Mirrors the final test, which is later in time and on unseen drugs / hospitals."""
    k = max(3, len(val_blocks))  # always hold out a third (or less) of units per fold
    groups = np.sort(df[group_col].unique())
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(groups, rng.permutation(len(groups)) % k))
    gfold = df[group_col].map(fold_of).values
    t = df[time_col].values
    folds = []
    for i, (start, end) in enumerate(val_blocks):
        val = np.where((gfold == i) & (t >= start) & (t <= end))[0]
        train = np.where((gfold != i) & (t <= gap(start)))[0]
        folds.append((train, val))
    return folds


def clf_metrics(y, p):
    return {
        "AUROC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "AUPRC": average_precision_score(y, p) if y.sum() > 0 else np.nan,
        "Brier": brier_score_loss(y, p),
    }


def reg_metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
    }


def cluster_bootstrap(df, clusters, metric_fn, n_boot=1000, seed=SEED):
    """Point estimate and percentile 95% CI, resampling whole clusters (hospitals or drug ingredients)."""
    rng = np.random.default_rng(seed)
    point = metric_fn(df)
    groups = df.groupby(clusters).indices
    keys = np.array(list(groups))
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        draws.append(metric_fn(df.iloc[idx]))
    draws = pd.DataFrame(draws)
    lo, hi = draws.quantile(0.025), draws.quantile(0.975)
    return {k: (point[k], lo[k], hi[k]) for k in point}


def fmt_ci(v, digits=3, scale=1.0):
    est, lo, hi = v
    return f"{est*scale:.{digits}f} ({lo*scale:.{digits}f} to {hi*scale:.{digits}f})"


def calibration_intercept_slope(y, p):
    """Logistic recalibration: logit(P(y)) = a + b * logit(p)."""
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return intercept, slope
