import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. DATA
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "datasets" / "Amine_in_water_processed.xlsx"
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
# 4. BASE MODEL 1: CatBoost (OPTIMIZED)
# ============================================================
catboost_model = CatBoostRegressor(
    learning_rate=0.15,
    iterations=700,
    depth=8,
    bagging_temperature=1,
    border_count=64,
    min_data_in_leaf=1,
    colsample_bylevel=0.55,
    random_state=42,
    verbose=False
)

# ============================================================
# 5. BASE MODEL 2: ExtraTrees (OPTIMIZED)
# ============================================================
extratrees_model = make_pipeline(
    ExtraTreesRegressor(
        n_estimators=250,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        random_state=42
    )
)

# ============================================================
# 6. BASE MODEL 3: SVR (OPTIMIZED)
# ============================================================
svr_model = make_pipeline(
    StandardScaler(),
    SVR(
        kernel="rbf",
        C=50,
        epsilon=1e-3,
        tol=1e-4,
        gamma="scale"
    )
)

# ============================================================
# 7. STACKING: CatBoost+ExtraTrees+SVR ============================================================
stacking_model = StackingRegressor(
    estimators=[
        ("et", extratrees_model),
        ("cb", catboost_model),
        ("svr", svr_model)
    ],
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=False,
    n_jobs=None
)

# ============================================================
# 8. 10-FOLD CROSS-VALIDATION (STACKING)
# ============================================================
cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    stacking_model,
    X_train,
    Y_train,
    cv=cv,
    scoring="r2",
    n_jobs=None
)

print("\n10-FOLD CV")
print(f"Mean R² = {np.mean(cv_scores):.6f}")
print(f"SD R²   = {np.std(cv_scores):.6f}")


# ============================================================
# 9. MODEL FIT
# ============================================================
stacking_model.fit(
    X_train,
    Y_train
)

# ============================================================
# 10. PREDICTION
# ============================================================
Y_pred_train = stacking_model.predict(
    X_train
)

Y_pred_test = stacking_model.predict(
    X_test
)

# ============================================================
# 11. PERFORMANCE METRICS
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


train_mse, train_rmse, train_mae, train_r2 = metrics(
    Y_train,
    Y_pred_train
)

test_mse, test_rmse, test_mae, test_r2 = metrics(
    Y_test,
    Y_pred_test
)

# ============================================================
# 12. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL STACKING MODEL: CatBoost+ExtraTrees+SVR")
print("META-MODEL: Linear regression")
print("=" * 60)

print("\nTRAINING SET")
print(f"MSE  = {train_mse:.8f}")
print(f"RMSE = {train_rmse:.8f}")
print(f"MAE  = {train_mae:.8f}")
print(f"R²   = {train_r2:.6f}")

print("\nTEST SET")
print(f"MSE  = {test_mse:.8f}")
print(f"RMSE = {test_rmse:.8f}")
print(f"MAE  = {test_mae:.8f}")
print(f"R²   = {test_r2:.6f}")


print("\nExecutioncompleted.")
input("Press ENTER to close...")
