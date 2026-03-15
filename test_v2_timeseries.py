"""
=================================================================
자금예측 ML — TimeSeriesSplit 교차검증 테스트 (test_v2_timeseries.py)
=================================================================
단일 train/valid 분할 대신 시계열 교차검증(expanding window)으로
모델 성능을 더 안정적으로 평가합니다.
=================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings('ignore')

from test import (
    CONFIG, load_data, prepare_monthly_target,
    create_features, get_models, evaluate_model
)


def train_with_tscv(config=None, n_splits=4, valid_size=6):
    """TimeSeriesSplit 교차검증으로 모델을 평가합니다.

    Expanding window 방식:
      Fold 1: train[0:N-18]  valid[N-18:N-12]
      Fold 2: train[0:N-12]  valid[N-12:N-6]
      Fold 3: train[0:N-6]   valid[N-6:N]
      ...
    """
    if config is None:
        config = CONFIG

    target = config["target_col"]

    print("=" * 60)
    print("  자금예측 ML — TimeSeriesSplit 교차검증 테스트")
    print("=" * 60)

    # 1) 데이터 준비
    print("\n[1] 데이터 준비")
    data = load_data(config)
    monthly = prepare_monthly_target(data, config)
    df_model, feature_cols = create_features(monthly, config)

    # 2) TimeSeriesSplit Fold 생성
    print(f"\n[2] TimeSeriesSplit (n_splits={n_splits}, valid_size={valid_size})")

    df_sorted = df_model.sort_values('date').reset_index(drop=True)
    total = len(df_sorted)

    folds = []
    for i in range(n_splits):
        valid_end_idx = total - i * valid_size
        valid_start_idx = valid_end_idx - valid_size
        train_end_idx = valid_start_idx

        if train_end_idx < 12:  # 최소 학습 데이터 12건
            break

        train_data = df_sorted.iloc[:train_end_idx]
        valid_data = df_sorted.iloc[valid_start_idx:valid_end_idx]

        folds.append({
            'fold': n_splits - i,
            'train': train_data,
            'valid': valid_data,
            'train_period': f"{train_data[config['date_col']].min()} ~ {train_data[config['date_col']].max()}",
            'valid_period': f"{valid_data[config['date_col']].min()} ~ {valid_data[config['date_col']].max()}",
        })

    folds = sorted(folds, key=lambda x: x['fold'])

    for fold in folds:
        print(f"  Fold {fold['fold']}: "
              f"학습 {len(fold['train'])}건 [{fold['train_period']}] | "
              f"검증 {len(fold['valid'])}건 [{fold['valid_period']}]")

    # 3) 모델별 교차검증
    print(f"\n[3] 모델 학습 & 교차검증 평가")

    model_names = list(get_models(config).keys())
    cv_results = {name: [] for name in model_names}

    for fold in folds:
        print(f"\n  --- Fold {fold['fold']} ---")

        X_train = fold['train'][feature_cols]
        y_train = fold['train'][target]
        X_valid = fold['valid'][feature_cols]
        y_valid = fold['valid'][target]

        models = get_models(config)

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
                cv_results[name].append({
                    'fold': fold['fold'],
                    'MAE': metrics['MAE'],
                    'MAPE': metrics['MAPE(%)'],
                    'RMSE': metrics['RMSE'],
                })
                print(f"    {name:>20s}: MAE={metrics['MAE']:>15,.0f}  "
                      f"MAPE={metrics['MAPE(%)']:>6.2f}%")

            except Exception as e:
                print(f"    {name:>20s}: ❌ {e}")

    # 4) 교차검증 평균 결과
    print("\n" + "=" * 70)
    print(f"  📊 TimeSeriesSplit 교차검증 평균 결과 ({len(folds)} folds)")
    print("=" * 70)

    summary = []
    for name in model_names:
        if cv_results[name]:
            maes = [r['MAE'] for r in cv_results[name]]
            mapes = [r['MAPE'] for r in cv_results[name]]
            rmses = [r['RMSE'] for r in cv_results[name]]
            summary.append({
                'Model': name,
                'Avg_MAE': np.mean(maes),
                'Std_MAE': np.std(maes),
                'Avg_MAPE': np.mean(mapes),
                'Avg_RMSE': np.mean(rmses),
                'Folds': len(maes),
            })

    summary_df = pd.DataFrame(summary).sort_values('Avg_MAE')

    print(f"\n  {'모델':>20s} | {'평균 MAE':>15s} | {'MAE 표준편차':>15s} | "
          f"{'평균 MAPE':>10s} | Folds")
    print("  " + "-" * 80)
    for _, row in summary_df.iterrows():
        print(f"  {row['Model']:>20s} | {row['Avg_MAE']:>15,.0f} | "
              f"{row['Std_MAE']:>15,.0f} | {row['Avg_MAPE']:>9.2f}% | "
              f"{row['Folds']:.0f}")

    # Fold별 상세
    print(f"\n  📋 Fold별 상세 (MAE)")
    print(f"  {'모델':>20s}", end="")
    for fold in folds:
        print(f" | {'Fold '+str(fold['fold']):>12s}", end="")
    print(f" | {'평균':>15s}")
    print("  " + "-" * (25 + 15 * (len(folds) + 1)))

    for _, row in summary_df.iterrows():
        name = row['Model']
        print(f"  {name:>20s}", end="")
        for fold_result in sorted(cv_results[name], key=lambda x: x['fold']):
            print(f" | {fold_result['MAE']:>12,.0f}", end="")
        print(f" | {row['Avg_MAE']:>15,.0f}")

    return summary_df, cv_results


def compare_with_single_split(config=None):
    """단일 분할 vs TimeSeriesSplit 비교."""
    if config is None:
        config = CONFIG

    target = config["target_col"]

    # --- 단일 분할 ---
    print("\n" + "=" * 70)
    print("  [A] 단일 분할 (현재 방식)")
    print("=" * 70)

    data = load_data(config)
    monthly = prepare_monthly_target(data, config)
    df_model, feature_cols = create_features(monthly, config)

    valid_months = config.get("valid_months", 6)
    cutoff = df_model['date'].max() - pd.DateOffset(months=valid_months - 1)
    train = df_model[df_model['date'] < cutoff]
    valid = df_model[df_model['date'] >= cutoff]

    X_train = train[feature_cols]
    y_train = train[target]
    X_valid = valid[feature_cols]
    y_valid = valid[target]

    models = get_models(config)
    single_results = []

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
            single_results.append(metrics)
        except:
            pass

    single_df = pd.DataFrame(single_results).sort_values("MAE")
    print("\n  단일 분할 MAE:")
    for _, r in single_df.iterrows():
        print(f"    {r['Model']:>20s}: MAE={r['MAE']:>15,.0f}  MAPE={r['MAPE(%)']:>6.2f}%")

    # --- TimeSeriesSplit ---
    print("\n" + "=" * 70)
    print("  [B] TimeSeriesSplit 교차검증")
    print("=" * 70)
    tscv_df, cv_results = train_with_tscv(config, n_splits=4, valid_size=6)

    # --- 비교 ---
    print("\n" + "=" * 70)
    print("  📊 단일 분할 vs TimeSeriesSplit 비교")
    print("=" * 70)
    print(f"\n  {'모델':>20s} | {'단일 MAE':>15s} | {'TSCV 평균 MAE':>15s} | "
          f"{'MAE 편차':>12s} | {'판정':>8s}")
    print("  " + "-" * 80)

    for _, tscv_row in tscv_df.iterrows():
        name = tscv_row['Model']
        single_row = single_df[single_df['Model'] == name]
        if len(single_row) > 0:
            s_mae = single_row.iloc[0]['MAE']
            t_mae = tscv_row['Avg_MAE']
            t_std = tscv_row['Std_MAE']
            diff = t_mae - s_mae
            # TSCV의 평균 MAE가 단일보다 낮으면 → 단일이 과대평가(낙관적)였을 수 있음
            note = "안정적" if t_std < s_mae * 0.3 else "불안정"
            print(f"  {name:>20s} | {s_mae:>15,.0f} | {t_mae:>15,.0f} | "
                  f"±{t_std:>10,.0f} | {note:>8s}")

    print()


if __name__ == "__main__":
    compare_with_single_split()
