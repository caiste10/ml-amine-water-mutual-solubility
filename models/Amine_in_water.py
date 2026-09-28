import numpy as np
import pandas as pd
import shap
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. DATA
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "datasets" / "amine_in_water_processed.xlsx"
Data = pd.read_excel(DATA_PATH)
X0 = Data.iloc[:, 0:9].values
Y0 = Data.iloc[:, 9].values
labels = Data.iloc[:, -1].values
temperature = Data.iloc[:, 0].values

# ============================================================
# 2. STRATIFICATION
# ============================================================
temperature_bins = pd.qcut(
    temperature,
    q=2,
    labels=False,
    duplicates="drop"
)

stratify_keys = np.array([
    f"{label}_{temp_bin}"
    for label, temp_bin in zip(labels, temperature_bins)
])

class_counts = Counter(stratify_keys)

valid_keys = {
    key
    for key, count in class_counts.items()
    if count > 1
}

mask_valid = np.array([
    key in valid_keys
    for key in stratify_keys
])

mask_excluded = ~mask_valid

X = X0[mask_valid]
Y = Y0[mask_valid]
stratify_keys_filtered = stratify_keys[mask_valid]

X_excluded = X0[mask_excluded]
Y_excluded = Y0[mask_excluded]

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42,
    stratify=stratify_keys_filtered
)

if len(X_excluded) > 0:
    X_train = np.vstack([
        X_train,
        X_excluded
    ])

    Y_train = np.concatenate([
        Y_train,
        Y_excluded
    ])

print(f"Training samples: {len(Y_train)}")
print(f"Test samples:     {len(Y_test)}")

# ============================================================
# 4. ExtraTrees MODEL (OPTIMIZED)
# ============================================================
extratrees_model = ExtraTreesRegressor(
    n_estimators=250,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=42
)

# ============================================================
# 5. 10-FOLD CROSS-VALIDATION
# ============================================================
cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    extratrees_model,
    X_train,
    Y_train,
    cv=cv,
    scoring="r2",
    n_jobs=None
)

print("\n10-FOLD CV - EXTRATREES")
print(f"Mean R² = {np.mean(cv_scores):.6f}")
print(f"SD R²   = {np.std(cv_scores):.6f}")

# ============================================================
# 6. MODEL FIT
# ============================================================
extratrees_model.fit(
    X_train,
    Y_train
)

# ============================================================
# 7. PREDICTION
# ============================================================
Y_pred_train = extratrees_model.predict(
    X_train
)

Y_pred_test = extratrees_model.predict(
    X_test
)

# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================
def metrics(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mse, rmse, mae, r2


def log_r2(y_true, y_pred, lower_bound):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    if np.any(y_true <= 0):
        raise ValueError(
            "Log-R² cannot be calculated because y_true contains values <= 0."
        )

    y_pred_for_log = np.where(
        y_pred <= 0,
        lower_bound,
        y_pred
    )

    return r2_score(
        np.log10(y_true),
        np.log10(y_pred_for_log)
    )


positive_training_values = Y_train[
    Y_train > 0
]

if len(positive_training_values) == 0:
    raise ValueError(
        "The training target does not contain positive values for Log-R²."
    )

log_r2_floor = np.min(
    positive_training_values
)

train_mse, train_rmse, train_mae, train_r2 = metrics(
    Y_train,
    Y_pred_train
)

test_mse, test_rmse, test_mae, test_r2 = metrics(
    Y_test,
    Y_pred_test
)

train_log_r2 = log_r2(
    Y_train,
    Y_pred_train,
    log_r2_floor
)

test_log_r2 = log_r2(
    Y_test,
    Y_pred_test,
    log_r2_floor
)

# ============================================================
# 9. FINAL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("FINAL MODEL: ExtraTrees")
print("=" * 60)

print("\nTRAINING SET")
print(f"MSE      = {train_mse:.8f}")
print(f"RMSE     = {train_rmse:.8f}")
print(f"MAE      = {train_mae:.8f}")
print(f"R²       = {train_r2:.6f}")
print(f"Log-R²   = {train_log_r2:.6f}")

print("\nTEST SET")
print(f"MSE      = {test_mse:.8f}")
print(f"RMSE     = {test_rmse:.8f}")
print(f"MAE      = {test_mae:.8f}")
print(f"R²       = {test_r2:.6f}")
print(f"Log-R²   = {test_log_r2:.6f}")


print("\nExecution completed.")
input("Press ENTER to close...")
