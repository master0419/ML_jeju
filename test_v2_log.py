"""
=================================================================
자금예측 ML — 로그 변환 테스트 (test_v2_log.py)
=================================================================
원본 test.py에서 타겟값에 log1p 변환 적용.
큰 값/작은 값의 오차를 균등하게 학습하여 변동이 큰 월의 예측 개선 검증.
=================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings('ignore')

# test.py에서 공통 함수/설정 import
from test import (
    CONFIG, load_data, prepare_monthly_target,
    create_features, get_models, evaluate_model
)


def train_and_evaluate_log(config=None):
    """로그 변환 적용 파이프라인."""
    if config is None:
        config = CONFIG

    print("=" * 60)
    print("  자금예측 ML — 로그 변환 테스트")
    print("=" * 60)

    target = config["target_col"]

    # 1) 데이터 준비
    print("\n[1] 데이터 준비")
    data = load_data(config)
    monthly = prepare_monthly_target(data, config)

    # 2) 로그 변환 전 원본 백업
    monthly_orig = monthly.copy()
    print(f"\n  📊 원본 타겟 범위: {monthly[target].min():,.0f} ~ {monthly[target].max():,.0f}")
    print(f"  📊 원본 타겟 표준편차: {monthly[target].std():,.0f}")

    # 3) 로그 변환 적용
    monthly[target] = np.log1p(monthly[target])
    print(f"  📊 로그 변환 후 범위: {monthly[target].min():.4f} ~ {monthly[target].max():.4f}")
    print(f"  📊 로그 변환 후 표준편차: {monthly[target].std():.4f}")

    # 4) 피처 엔지니어링 (로그 공간)
    print("\n[2] 피처 엔지니어링 (로그 공간)")
    df_model, feature_cols = create_features(monthly, config)

    # 5) 학습/검증 분할
    print("\n[3] 학습/검증 분할")
    valid_months = config.get("valid_months", 6)
    cutoff = df_model['date'].max() - pd.DateOffset(months=valid_months - 1)
    train = df_model[df_model['date'] < cutoff]
    valid = df_model[df_model['date'] >= cutoff]

    X_train = train[feature_cols]
    y_train_log = train[target]
    X_valid = valid[feature_cols]
    y_valid_log = valid[target]

    # 원본 스케일의 검증 실제값
    y_valid_orig = np.expm1(y_valid_log.values)

    print(f"  ✓ 학습셋: {len(train)}건")
    print(f"  ✓ 검증셋: {len(valid)}건")

    # 6) 모델 학습
    print("\n[4] 모델 학습 & 평가")
    models = get_models(config)
    results_log = []
    results_orig = []

    for name, model in models.items():
        print(f"\n  🔄 {name} 학습 중...")
        try:
            start = time.time()
            patience = config.get("early_stopping_rounds", 30)

            if name == "XGBoost":
                model.set_params(early_stopping_rounds=patience)
                model.fit(X_train, y_train_log,
                          eval_set=[(X_valid, y_valid_log)], verbose=False)
            elif name == "LightGBM":
                from lightgbm import early_stopping as lgb_es
                model.fit(X_train, y_train_log,
                          eval_set=[(X_valid, y_valid_log)],
                          callbacks=[lgb_es(stopping_rounds=patience, verbose=False)])
            else:
                model.fit(X_train, y_train_log)

            elapsed = time.time() - start

            # 로그 공간 예측
            y_pred_log = model.predict(X_valid)

            # 원본 스케일로 역변환
            y_pred_orig = np.expm1(y_pred_log)

            # 원본 스케일에서 평가 (공정 비교)
            metrics = evaluate_model(y_valid_orig, y_pred_orig, model_name=name)
            metrics["Time(s)"] = round(elapsed, 3)
            results_orig.append(metrics)

            print(f"     ✅ MAE={metrics['MAE']:,.0f}, "
                  f"MAPE={metrics['MAPE(%)']}%, "
                  f"RMSE={metrics['RMSE']:,.0f}")

        except Exception as e:
            print(f"     ❌ 실패: {e}")

    # 7) 결과 비교
    results_df = pd.DataFrame(results_orig).sort_values("MAE", ascending=True)

    print("\n" + "=" * 70)
    print("  📊 로그 변환 모델 비교 (원본 스케일 MAE 기준)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return results_df


def compare_with_original(config=None):
    """원본(비변환) vs 로그 변환 결과를 비교합니다."""
    if config is None:
        config = CONFIG

    target = config["target_col"]

    # --- 원본 (비변환) ---
    print("\n" + "=" * 70)
    print("  [A] 원본 (변환 없음)")
    print("=" * 70)

    data = load_data(config)
    monthly_orig = prepare_monthly_target(data, config)
    df_model, feature_cols = create_features(monthly_orig, config)

    valid_months = config.get("valid_months", 6)
    cutoff = df_model['date'].max() - pd.DateOffset(months=valid_months - 1)
    train = df_model[df_model['date'] < cutoff]
    valid = df_model[df_model['date'] >= cutoff]

    X_train = train[feature_cols]
    y_train = train[target]
    X_valid = valid[feature_cols]
    y_valid = valid[target]

    models = get_models(config)
    orig_results = []

    for name, model in models.items():
        try:
            patience = config.get("early_stopping_rounds", 30)
            if name == "XGBoost":
                model.set_params(early_stopping_rounds=patience)
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)], verbose=False)
            elif name == "LightGBM":
                from lightgbm import early_stopping as lgb_es
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)],
                          callbacks=[lgb_es(stopping_rounds=patience, verbose=False)])
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            metrics = evaluate_model(y_valid.values, y_pred, model_name=name)
            orig_results.append(metrics)
        except:
            pass

    orig_df = pd.DataFrame(orig_results).sort_values("MAE")
    print("\n  원본 MAE:")
    for _, r in orig_df.iterrows():
        print(f"    {r['Model']:>20s}: MAE={r['MAE']:>15,.0f}  MAPE={r['MAPE(%)']:>6.2f}%")

    # --- 로그 변환 ---
    print("\n" + "=" * 70)
    print("  [B] 로그 변환")
    print("=" * 70)
    log_df = train_and_evaluate_log(config)

    # --- 비교 ---
    print("\n" + "=" * 70)
    print("  📊 원본 vs 로그 변환 비교")
    print("=" * 70)
    print(f"\n  {'모델':>20s} | {'원본 MAE':>15s} | {'로그 MAE':>15s} | {'변화':>10s}")
    print("  " + "-" * 70)

    for _, orig_row in orig_df.iterrows():
        name = orig_row["Model"]
        orig_mae = orig_row["MAE"]
        log_row = log_df[log_df["Model"] == name]
        if len(log_row) > 0:
            log_mae = log_row.iloc[0]["MAE"]
            diff = log_mae - orig_mae
            pct = (diff / orig_mae * 100) if orig_mae != 0 else 0
            marker = "✅ 개선" if diff < 0 else "❌ 악화"
            print(f"  {name:>20s} | {orig_mae:>15,.0f} | {log_mae:>15,.0f} | "
                  f"{pct:>+6.1f}% {marker}")

    print()


if __name__ == "__main__":
    compare_with_original()
