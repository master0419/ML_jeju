"""
하이퍼파라미터 탐색 스크립트
각 모델별로 주요 HP 조합을 Grid Search 방식으로 탐색합니다.
"""
import itertools
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# test.py에서 데이터 준비 함수들을 import
from test import (
    CONFIG, load_data, prepare_monthly_target,
    create_features, split_timeseries, evaluate_model
)

# ============================================================
# 1. 데이터 준비 (1회만)
# ============================================================
print("=" * 60)
print("  하이퍼파라미터 탐색")
print("=" * 60)

print("\n[1] 데이터 준비")
tables = load_data(CONFIG)
monthly = prepare_monthly_target(tables, CONFIG)
df_model, feature_cols = create_features(monthly, CONFIG)
X_train, y_train, X_valid, y_valid = split_timeseries(df_model, feature_cols, CONFIG)

# ============================================================
# 2. 탐색할 HP 그리드 정의
# ============================================================

# --- XGBoost ---
xgb_grid = {
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "reg_alpha":        [0.1, 0.5, 1.0],
    "reg_lambda":       [1.0, 2.0, 3.0],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

# --- LightGBM ---
lgb_grid = {
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "num_leaves":       [10, 15, 20, 31],
    "reg_alpha":        [0.1, 0.5, 1.0],
    "reg_lambda":       [1.0, 2.0, 3.0],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

# --- GradientBoosting ---
gbm_grid = {
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "n_estimators":     [100, 200, 300],
}

# --- RandomForest ---
rf_grid = {
    "max_depth":        [4, 5, 6, 8, 10],
    "n_estimators":     [100, 200, 300],
}


# ============================================================
# 3. 랜덤 샘플링 탐색 (조합이 너무 많으므로)
# ============================================================

def grid_to_samples(grid, max_samples=200):
    """전체 그리드에서 랜덤 샘플링 (조합 폭발 방지)."""
    keys = list(grid.keys())
    values = list(grid.values())
    all_combos = list(itertools.product(*values))
    
    total = len(all_combos)
    if total <= max_samples:
        samples = all_combos
    else:
        rng = np.random.RandomState(42)
        idx = rng.choice(total, max_samples, replace=False)
        samples = [all_combos[i] for i in idx]
    
    return [dict(zip(keys, combo)) for combo in samples], total


def search_model(model_name, grid, base_params, create_fn, max_samples=200):
    """특정 모델의 HP를 탐색합니다."""
    combos, total = grid_to_samples(grid, max_samples)
    print(f"\n{'='*60}")
    print(f"  {model_name}: 전체 {total}개 조합 중 {len(combos)}개 탐색")
    print(f"{'='*60}")
    
    results = []
    best_mae = float('inf')
    best_params = None
    patience = CONFIG.get("early_stopping_rounds", 30)
    
    for i, hp in enumerate(combos):
        params = {**base_params, **hp}
        try:
            model = create_fn(**params)
            
            if model_name == "XGBoost":
                model.set_params(early_stopping_rounds=patience)
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)], verbose=False)
            elif model_name == "LightGBM":
                from lightgbm import early_stopping as lgb_es
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)],
                          callbacks=[lgb_es(stopping_rounds=patience, verbose=False)])
            else:
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_valid)
            metrics = evaluate_model(y_valid.values, y_pred, model_name)
            mae = metrics["MAE"]
            mape = metrics["MAPE(%)"]
            
            results.append({**hp, "MAE": mae, "MAPE": mape})
            
            if mae < best_mae:
                best_mae = mae
                best_params = hp.copy()
                print(f"  [{i+1}/{len(combos)}] ★ 신규 최적! MAE={mae:,.0f} "
                      f"MAPE={mape}% | {hp}")
            
            if (i + 1) % 50 == 0 and mae >= best_mae:
                print(f"  [{i+1}/{len(combos)}] 진행 중... 현재 최적 MAE={best_mae:,.0f}")
                
        except Exception as e:
            pass
    
    print(f"\n  🏆 {model_name} 최적 HP:")
    print(f"     MAE = {best_mae:,.0f}")
    for k, v in best_params.items():
        print(f"     {k} = {v}")
    
    return best_params, best_mae, pd.DataFrame(results)


# ============================================================
# 4. 각 모델 탐색 실행
# ============================================================

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

all_best = {}

# --- XGBoost ---
xgb_base = {"n_estimators": 300, "random_state": 42, "verbosity": 0}
best_hp, best_mae, df_res = search_model(
    "XGBoost", xgb_grid, xgb_base,
    lambda **p: XGBRegressor(**p),
    max_samples=200
)
all_best["XGBoost"] = {"params": {**xgb_base, **best_hp}, "MAE": best_mae}

# --- LightGBM ---
lgb_base = {"n_estimators": 300, "random_state": 42, "verbose": -1}
best_hp, best_mae, df_res = search_model(
    "LightGBM", lgb_grid, lgb_base,
    lambda **p: LGBMRegressor(**p),
    max_samples=200
)
all_best["LightGBM"] = {"params": {**lgb_base, **best_hp}, "MAE": best_mae}

# --- GradientBoosting ---
gbm_base = {"random_state": 42}
best_hp, best_mae, df_res = search_model(
    "GradientBoosting", gbm_grid, gbm_base,
    lambda **p: GradientBoostingRegressor(**p),
    max_samples=200
)
all_best["GradientBoosting"] = {"params": {**gbm_base, **best_hp}, "MAE": best_mae}

# --- RandomForest ---
rf_base = {"random_state": 42, "n_jobs": -1}
best_hp, best_mae, df_res = search_model(
    "RandomForest", rf_grid, rf_base,
    lambda **p: RandomForestRegressor(**p),
    max_samples=200
)
all_best["RandomForest"] = {"params": {**rf_base, **best_hp}, "MAE": best_mae}


# ============================================================
# 5. 최종 요약
# ============================================================
print("\n" + "=" * 60)
print("  🏆 최종 최적 하이퍼파라미터 요약")
print("=" * 60)

for name, info in all_best.items():
    print(f"\n  [{name}] MAE = {info['MAE']:,.0f}")
    for k, v in info["params"].items():
        print(f"    {k}: {v}")

# JSON으로 저장
import json
with open("./result/best_hyperparams.json", "w", encoding="utf-8") as f:
    # float/int 변환
    save_data = {}
    for name, info in all_best.items():
        save_data[name] = {
            "MAE": float(info["MAE"]),
            "params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in info["params"].items()}
        }
    json.dump(save_data, f, indent=2, ensure_ascii=False)

print(f"\n  ✓ 최적 HP 저장: ./result/best_hyperparams.json")
print("\n✅ 탐색 완료!")
