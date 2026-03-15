"""
=================================================================
자금예측 ML 테스트 환경 (v2 — 앙상블 융합 지원)
=================================================================
- 개별 모델 + 앙상블(Voting, Stacking) 융합 지원
- XGBoost + LightGBM + GradientBoosting 조합
- 시계열 재귀적 예측 (recursive forecasting)
- test.ipynb에서 import하여 연동 사용 가능

[사용법]
  A) 직접 실행:  python test.py
  B) 노트북 연동: from test import CONFIG, main
=================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
import time
import hashlib
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================
# [CONFIG] 여기만 수정하면 됩니다!
# ============================================================

CONFIG = {
    # ---- 데이터 ----
    "data_dir": r"C:\Users\7slwm\Downloads\data_jeju\data",
    "data_files": {
        "acc_pay":  "acc_pay_df.parquet",   # df1: 매입채무
        "acc_rec":  "acc_rec_df.parquet",   # df2: 매출채권
        "ap_days":  "ap_days_df.parquet",   # df3: 매입기일
        "ar_days":  "ar_days_df.parquet",   # df4: 매출기일
        "coa":      "coa_df.parquet",       # df5: 계정과목
        "purchase": "purchase_df.parquet",  # df6: 매입
        "revenue":  "revenue_df.parquet",   # df7: 매출
    },

    # ---- 타겟 변수 ----
    "target_col": "TOT_MN_MNAM",       # 총공급가액 (월별 합계 예측)

    # ---- 데이터 설정 ----
    "main_table": "revenue",           # 메인 테이블 (data_files 키)

    # ---- 시계열 설정 ----
    "date_col": "DM_DATA",            # 날짜 컬럼
    "lag_periods": [1, 2, 3, 6, 12],  # 래그 변수 기간
    "rolling_windows": [3, 6, 12],    # 이동평균 윈도우
    "forecast_months": 6,             # 미래 예측 개월수
    "forecast_start": "2025-02",      # 예측 시작월

    # ---- 학습 설정 ----
    "valid_months": 6,                # 검증셋 개월수
    "random_state": 42,
    "early_stopping_rounds": 30,      # patience: 30라운드 개선 없으면 학습 중단

    # ---- 사용할 모델 (True/False로 ON/OFF) ----
    "models": {
        "XGBoost":            True,
        "LightGBM":           True,
        "GradientBoosting":   True,
        "RandomForest":       True,
        "Ridge":              True,
        # 앙상블 융합 모델
        "Ensemble_Voting":    True,    # XGB + LGB + GBM 가중평균
        "Ensemble_Stacking":  True,    # XGB + LGB + GBM → Ridge 메타모델
        # 비활성화 모델
        "LinearRegression":   False,
        "Lasso":              False,
        "DecisionTree":       False,
        "SVR":                False,
        "KNN":                False,
    },

    # ---- 모델 하이퍼파라미터 ----
    "model_params": {
        "XGBoost": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": 42, "verbosity": 0,
        },
        "LightGBM": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": 42, "verbose": -1,
        },
        "GradientBoosting": {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "random_state": 42,
        },
        "RandomForest": {
            "n_estimators": 200, "max_depth": 10,
            "random_state": 42, "n_jobs": -1,
        },
        "Ridge":          {"alpha": 1.0},
        "Lasso":          {"alpha": 1.0},
        "DecisionTree":   {"max_depth": 10, "random_state": 42},
        "SVR":            {"kernel": "rbf", "C": 1.0},
        "KNN":            {"n_neighbors": 5},
    },

    # ---- 앙상블 융합 설정 ----
    "ensemble": {
        "voting_weights": [2, 2, 1],   # XGB, LGB, GBM 가중치
        "stacking_final": "Ridge",     # 메타 모델
        "stacking_cv": 3,             # 교차검증 fold
    },

    # ---- 모델 저장 설정 ----
    "model_dir": "./model",               # 최적 모델 저장 폴더
    "history_dir": "./model/history",      # 학습 이력 폴더

    # ---- 출력 설정 ----
    "save_results": True,
    "results_dir": "./result",
    "plot_results": True,
}


# ============================================================
# 컬럼명 한글 → 영어 매핑
# ============================================================

RENAME_MAP = {
    # 공통
    '기준년월': 'DW_BAS_NYYMM',
    '사업자등록번호': 'NO_BIZ',
    '법인등록번호': 'NO_CORPOR',
    '거래비중': 'TRAN_RATE',
    # df1 (매입채무)
    '매입처사업자번호': 'NO_BISOCIAL',
    '매입처명': 'NM_TRADE',
    '발생(지급)월': 'DM_DATA',
    '매입채무발생액': 'MN_AP',
    '매입채무지급액': 'MN_AP_PAY',
    '매입채무잔액': 'MN_AP_BAL',
    '총매입채무발생액': 'TOT_MN_AP',
    '총매입채무지급액': 'TOT_MN_AP_PAY',
    '총매입채무잔액': 'TOT_MN_AP_BAL',
    # df2 (매출채권)
    '매출처사업자번호': 'NO_BISOCIAL',
    '매출처명': 'NM_TRADE',
    '발생(회수)월': 'DM_DATA',
    '매출채권발생액': 'MN_AR',
    '매출채권회수액': 'MN_AR_RCV',
    '매출채권잔액': 'MN_AR_BAL',
    '총매출채권발생액': 'TOT_MN_AR',
    '총매출채권회수액': 'TOT_MN_AR_RCV',
    '총매출채권잔액': 'TOT_MN_AR_BAL',
    # df3 (매입채무지급기간)
    '기간구분': 'CD_TERM',
    '매입채무지급기간': 'AP_DAYS',
    '가중평균지급기간': 'AP_DAYS_W',
    # df4 (매출채권회수기간)
    '매출채권회수기간': 'AR_DAYS',
    '가중평균회수기간': 'AR_DAYS_W',
    # df5/df6/df7 공통
    '공급시기': 'DM_DATA',
    '공급가액': 'MN_MNAM',
    '부가세': 'MN_VAT',
    '매출유형': 'TY_MTH2',
    '매입유형': 'TY_MTH2',
    '공급건수': 'CT_MNAM',
    '취소건수': 'CT_MNAM_M',
    '취소금액': 'MN_MNAM_M',
    '총공급가액': 'TOT_MN_MNAM',
    '총취소금액': 'TOT_MN_MNAM_M',
}


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_data(config=None):
    """parquet 파일들을 로드하고 컬럼명을 영어로 변환합니다."""
    if config is None:
        config = CONFIG
    data = {}
    for name, filename in config["data_files"].items():
        filepath = os.path.join(config["data_dir"], filename)
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            df = df.rename(columns=RENAME_MAP)
            data[name] = df
            print(f"  ✓ {name}: {df.shape}")
        else:
            print(f"  ✗ {name}: 파일 없음 ({filepath})")
    return data


# ============================================================
# 2. 월별 타겟 시계열 생성
# ============================================================

def prepare_monthly_target(data, config=None):
    """메인 테이블에서 월별 타겟을 집계합니다.

    사업자별(NO_BIZ) 월별(DM_DATA) 고유값을 추출 후 전체 합산.
    """
    if config is None:
        config = CONFIG
    main_key = config["main_table"]
    target = config["target_col"]
    date_col = config["date_col"]

    df = data[main_key].copy()
    monthly = (
        df.groupby(['NO_BIZ', date_col])[target]
        .first()
        .reset_index()
        .groupby(date_col)[target]
        .sum()
        .reset_index()
    )
    monthly.columns = [date_col, target]
    monthly[date_col] = monthly[date_col].astype(str)
    monthly = monthly.sort_values(date_col).reset_index(drop=True)
    monthly['date'] = pd.to_datetime(monthly[date_col])

    print(f"  ✓ 월별 시계열: {len(monthly)}건 "
          f"({monthly[date_col].min()} ~ {monthly[date_col].max()})")
    return monthly


# ============================================================
# 3. 피처 엔지니어링
# ============================================================

def create_features(monthly, config=None):
    """시계열 피처를 생성합니다."""
    if config is None:
        config = CONFIG
    target = config["target_col"]
    df = monthly.copy()

    # 시간 피처
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # 라그 피처
    for lag in config["lag_periods"]:
        df[f'lag_{lag}'] = df[target].shift(lag)

    # 이동평균 / 이동표준편차
    for w in config["rolling_windows"]:
        df[f'rolling_mean_{w}'] = df[target].rolling(w).mean()
        df[f'rolling_std_{w}'] = df[target].rolling(w).std()

    # 변화율
    df['pct_change_1'] = df[target].pct_change(1)
    df['pct_change_12'] = df[target].pct_change(12)

    # 피처 컬럼 목록
    feature_cols = [c for c in df.columns
                    if c not in [config["date_col"], 'date', target]]

    # NaN 제거
    df_model = df.dropna(subset=feature_cols).copy()

    print(f"  ✓ 피처 {len(feature_cols)}개: {feature_cols}")
    print(f"  ✓ 학습 가능 데이터: {len(df_model)}건 "
          f"({df_model[config['date_col']].min()} ~ "
          f"{df_model[config['date_col']].max()})")
    return df_model, feature_cols


# ============================================================
# 4. 모델 정의 (앙상블 융합 포함)
# ============================================================

def get_models(config=None):
    """활성화된 모델들을 반환합니다 (앙상블 포함)."""
    if config is None:
        config = CONFIG
    model_map = {}
    params = config["model_params"]

    # --- 개별 모델 ---
    if config["models"].get("LinearRegression"):
        from sklearn.linear_model import LinearRegression
        model_map["LinearRegression"] = LinearRegression()

    if config["models"].get("Ridge"):
        from sklearn.linear_model import Ridge
        model_map["Ridge"] = Ridge(**params.get("Ridge", {}))

    if config["models"].get("Lasso"):
        from sklearn.linear_model import Lasso
        model_map["Lasso"] = Lasso(**params.get("Lasso", {}))

    if config["models"].get("DecisionTree"):
        from sklearn.tree import DecisionTreeRegressor
        model_map["DecisionTree"] = DecisionTreeRegressor(
            **params.get("DecisionTree", {}))

    if config["models"].get("RandomForest"):
        from sklearn.ensemble import RandomForestRegressor
        model_map["RandomForest"] = RandomForestRegressor(
            **params.get("RandomForest", {}))

    if config["models"].get("GradientBoosting"):
        from sklearn.ensemble import GradientBoostingRegressor
        model_map["GradientBoosting"] = GradientBoostingRegressor(
            **params.get("GradientBoosting", {}))

    if config["models"].get("XGBoost"):
        try:
            from xgboost import XGBRegressor
            model_map["XGBoost"] = XGBRegressor(**params.get("XGBoost", {}))
        except ImportError:
            print("  ⚠ XGBoost 미설치 → pip install xgboost")

    if config["models"].get("LightGBM"):
        try:
            from lightgbm import LGBMRegressor
            model_map["LightGBM"] = LGBMRegressor(
                **params.get("LightGBM", {}))
        except ImportError:
            print("  ⚠ LightGBM 미설치 → pip install lightgbm")

    if config["models"].get("SVR"):
        from sklearn.svm import SVR
        model_map["SVR"] = SVR(**params.get("SVR", {}))

    if config["models"].get("KNN"):
        from sklearn.neighbors import KNeighborsRegressor
        model_map["KNN"] = KNeighborsRegressor(**params.get("KNN", {}))

    # --- 앙상블 융합 모델 ---
    need_voting = config["models"].get("Ensemble_Voting")
    need_stacking = config["models"].get("Ensemble_Stacking")

    if need_voting or need_stacking:
        from sklearn.ensemble import VotingRegressor, StackingRegressor

        # 앙상블 기본 구성원 (별도 인스턴스)
        base_estimators = []
        try:
            from xgboost import XGBRegressor
            base_estimators.append(
                ('xgb', XGBRegressor(**params.get("XGBoost", {}))))
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor
            base_estimators.append(
                ('lgb', LGBMRegressor(**params.get("LightGBM", {}))))
        except ImportError:
            pass
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        base_estimators.append(
            ('gb', GBR(**params.get("GradientBoosting", {}))))

        if len(base_estimators) >= 2:
            ens_cfg = config.get("ensemble", {})

            if need_voting:
                weights = ens_cfg.get("voting_weights", None)
                if weights and len(weights) != len(base_estimators):
                    weights = None
                model_map["Ensemble_Voting"] = VotingRegressor(
                    estimators=base_estimators,
                    weights=weights,
                )

            if need_stacking:
                from sklearn.linear_model import Ridge
                model_map["Ensemble_Stacking"] = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=Ridge(alpha=1.0),
                    cv=ens_cfg.get("stacking_cv", 3),
                )
        else:
            print("  ⚠ 앙상블 기본 모델 부족 (최소 2개 필요)")

    print(f"  ✓ 활성 모델 {len(model_map)}개: {list(model_map.keys())}")
    return model_map


# ============================================================
# 5. 학습 / 검증 분할
# ============================================================

def split_timeseries(df_model, feature_cols, config=None):
    """시계열 데이터를 학습/검증으로 분할합니다."""
    if config is None:
        config = CONFIG
    target = config["target_col"]
    valid_months = config.get("valid_months", 6)

    cutoff = df_model['date'].max() - pd.DateOffset(months=valid_months - 1)
    train = df_model[df_model['date'] < cutoff]
    valid = df_model[df_model['date'] >= cutoff]

    X_train = train[feature_cols]
    y_train = train[target]
    X_valid = valid[feature_cols]
    y_valid = valid[target]

    print(f"  ✓ 학습셋: {len(train)}건 "
          f"({train[config['date_col']].min()} ~ "
          f"{train[config['date_col']].max()})")
    print(f"  ✓ 검증셋: {len(valid)}건 "
          f"({valid[config['date_col']].min()} ~ "
          f"{valid[config['date_col']].max()})")
    return X_train, y_train, X_valid, y_valid


# ============================================================
# 6. 평가 지표
# ============================================================

def evaluate_model(y_true, y_pred, model_name=""):
    """회귀 평가지표를 계산합니다 (MAE, MAPE, MSE, RMSE)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = y_true != 0
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.sum() > 0 else np.nan)

    return {
        "Model": model_name,
        "MAE": round(mae, 0),
        "MAPE(%)": round(mape, 2),
        "MSE": round(mse, 0),
        "RMSE": round(rmse, 0),
    }


# ============================================================
# 7. 모델 학습 & 비교
# ============================================================

def train_all_models(models, X_train, y_train, X_valid, y_valid, config=None):
    """모든 모델을 학습하고 평가합니다."""
    if config is None:
        config = CONFIG
    results = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  🔄 {name} 학습 중...")
        try:
            start = time.time()

            patience = config.get("early_stopping_rounds", 30)

            if name == "XGBoost":
                model.set_params(early_stopping_rounds=patience)
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)], verbose=False)
                best_iter = getattr(model, 'best_iteration', None)
                if best_iter is not None:
                    print(f"     ⏱ 조기 중단: {best_iter}/{model.n_estimators} 라운드 "
                          f"(patience={patience})")
            elif name == "LightGBM":
                from lightgbm import early_stopping as lgb_early_stopping
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, y_valid)],
                          callbacks=[lgb_early_stopping(
                              stopping_rounds=patience, verbose=False)])
                best_iter = getattr(model, 'best_iteration_', None)
                if best_iter is not None:
                    print(f"     ⏱ 조기 중단: {best_iter}/{model.n_estimators} 라운드 "
                          f"(patience={patience})")
            else:
                model.fit(X_train, y_train)

            elapsed = time.time() - start
            y_pred = model.predict(X_valid)

            metrics = evaluate_model(y_valid.values, y_pred, model_name=name)
            metrics["Time(s)"] = round(elapsed, 3)

            results.append(metrics)
            predictions[name] = y_pred
            trained_models[name] = model

            print(f"     ✅ MAE={metrics['MAE']:,.0f}, "
                  f"MAPE={metrics['MAPE(%)']}%, "
                  f"MSE={metrics['MSE']:,.0f}, "
                  f"RMSE={metrics['RMSE']:,.0f}, "
                  f"시간={metrics['Time(s)']}s")

        except Exception as e:
            print(f"     ❌ 실패: {e}")
            results.append({
                "Model": name, "MAE": None, "RMSE": None,
                "MAPE(%)": None, "R²": None, "Error": str(e),
            })

    return results, predictions, trained_models


# ============================================================
# 8. 재귀적 미래 예측 (Recursive Forecasting)
# ============================================================

def recursive_forecast(model, monthly, feature_cols, config=None):
    """학습된 모델로 미래 N개월을 재귀적으로 예측합니다."""
    if config is None:
        config = CONFIG
    target = config["target_col"]
    forecast_months = config["forecast_months"]
    forecast_start = config["forecast_start"]

    future_dates = pd.date_range(forecast_start, periods=forecast_months, freq='MS')
    history = monthly[['date', target]].copy()
    preds = []

    for tgt_date in future_dates:
        row = {
            'year': tgt_date.year,
            'month': tgt_date.month,
            'quarter': (tgt_date.month - 1) // 3 + 1,
        }

        # 라그 피처
        for lag in config["lag_periods"]:
            lag_date = tgt_date - pd.DateOffset(months=lag)
            match = history[history['date'] == lag_date]
            row[f'lag_{lag}'] = (match[target].values[0]
                                if len(match) > 0 else np.nan)

        # 이동평균 / 표준편차
        recent = history.sort_values('date')[target].values
        for w in config["rolling_windows"]:
            if len(recent) >= w:
                row[f'rolling_mean_{w}'] = np.mean(recent[-w:])
                row[f'rolling_std_{w}'] = np.std(recent[-w:])
            else:
                row[f'rolling_mean_{w}'] = np.nan
                row[f'rolling_std_{w}'] = np.nan

        # 변화율
        if len(recent) >= 2 and recent[-2] != 0:
            row['pct_change_1'] = (recent[-1] - recent[-2]) / recent[-2]
        else:
            row['pct_change_1'] = 0

        lag12_match = history[
            history['date'] == tgt_date - pd.DateOffset(months=12)]
        if len(lag12_match) > 0 and lag12_match[target].values[0] != 0:
            row['pct_change_12'] = (
                (recent[-1] - lag12_match[target].values[0])
                / lag12_match[target].values[0])
        else:
            row['pct_change_12'] = 0

        # 예측
        X_future = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X_future)[0]
        preds.append({
            'date': tgt_date,
            config["date_col"]: tgt_date.strftime('%Y-%m'),
            f'predicted_{target}': pred,
        })

        # 예측값을 history에 추가 (다음 달 라그에 사용)
        history = pd.concat(
            [history, pd.DataFrame([{'date': tgt_date, target: pred}])],
            ignore_index=True)

    return pd.DataFrame(preds)


# ============================================================
# 9. 시각화
# ============================================================

def plot_model_comparison(results_df, monthly, pred_dfs,
                          y_valid, y_preds_valid, config=None):
    """4패널 비교 차트를 생성합니다."""
    if config is None:
        config = CONFIG

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    target = config["target_col"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- (1) 모델별 MAE 비교 (핵심 지표) ---
    ax = axes[0, 0]
    valid_mae = results_df.dropna(subset=["MAE"]).sort_values("MAE", ascending=False)
    best_mae = valid_mae["MAE"].min()
    colors = ['#2ecc71' if v == best_mae else '#3498db' for v in valid_mae["MAE"]]
    bars = ax.barh(valid_mae["Model"], valid_mae["MAE"], color=colors)
    for bar, v in zip(bars, valid_mae["MAE"]):
        ax.text(bar.get_width() + bar.get_width() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{v:,.0f}', va='center', fontsize=9)
    ax.set_xlabel("MAE")
    ax.set_title("모델별 MAE 비교 ★ (낮을수록 좋음)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- (2) 모델별 MAPE 비교 ---
    ax = axes[0, 1]
    valid_mape = results_df.dropna(subset=["MAPE(%)"]).sort_values(
        "MAPE(%)", ascending=False)
    colors = ['#e74c3c' if v > 20 else '#f39c12' if v > 10
              else '#2ecc71' for v in valid_mape["MAPE(%)"]]
    bars = ax.barh(valid_mape["Model"], valid_mape["MAPE(%)"], color=colors)
    for bar, v in zip(bars, valid_mape["MAPE(%)"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f}%', va='center', fontsize=9)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("모델별 MAPE 비교 (낮을수록 좋음)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- (3) 미래 예측 비교 (모델별) ---
    ax = axes[1, 0]
    recent = monthly[monthly['date'] >=
                     pd.Timestamp(config["forecast_start"])
                     - pd.DateOffset(months=12)]
    ax.plot(recent['date'], recent[target], 'k-o', markersize=4,
            label='실제값', alpha=0.8)

    palette = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
               '#e67e22', '#1abc9c', '#d35400']
    for i, (name, pdf) in enumerate(pred_dfs.items()):
        color = palette[i % len(palette)]
        ax.plot(pdf['date'], pdf[f'predicted_{target}'], '--s',
                color=color, markersize=5, label=name,
                alpha=0.8, linewidth=1.5)

    ax.axvline(x=pd.Timestamp(config["forecast_start"]),
               color='gray', linestyle='--', alpha=0.5)
    ax.set_title('미래 예측 비교 (모델별)', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x / 1e8:.0f}억'))
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # --- (4) 검증셋 실제 vs 예측 (MAE 최고 모델) ---
    ax = axes[1, 1]
    best_row = results_df.dropna(subset=["MAE"]).sort_values(
        "MAE", ascending=True).iloc[0]
    best_name = best_row["Model"]
    best_pred = y_preds_valid[best_name]
    x = range(len(y_valid))
    ax.bar(x, y_valid.values, alpha=0.6, label='실제값',
           color='steelblue', width=0.4)
    ax.bar([i + 0.4 for i in x], best_pred, alpha=0.6,
           label=f'{best_name} 예측', color='coral', width=0.4)
    ax.set_title(f"검증셋: {best_name} "
                 f"(MAE={best_row['MAE']:,.0f}, MAPE={best_row['MAPE(%)']:.1f}%)",
                 fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x / 1e8:.0f}억'))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if config["save_results"]:
        os.makedirs(config["results_dir"], exist_ok=True)
        path = os.path.join(config["results_dir"], "ensemble_forecast.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ 비교 차트 저장: {path}")

    plt.show()


def plot_feature_importance(trained_models, feature_cols,
                            config=None, top_n=10):
    """트리 기반 모델들의 피처 중요도를 비교합니다."""
    if config is None:
        config = CONFIG

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    tree_models = {n: m for n, m in trained_models.items()
                   if hasattr(m, 'feature_importances_')}
    if not tree_models:
        print("  ⚠ 피처 중요도 추출 가능한 모델 없음")
        return

    n = min(len(tree_models), 4)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, list(tree_models.items())[:4]):
        imp = pd.Series(model.feature_importances_,
                        index=feature_cols).sort_values(ascending=True)
        imp.tail(top_n).plot(kind='barh', ax=ax, color='teal')
        ax.set_title(f'{name} 피처 중요도', fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance')

    plt.tight_layout()

    if config["save_results"]:
        os.makedirs(config["results_dir"], exist_ok=True)
        path = os.path.join(config["results_dir"], "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 피처 중요도 저장: {path}")

    plt.show()


# ============================================================
# 10. 결과 저장
# ============================================================

def _compute_data_version(monthly, feature_cols, config):
    """데이터 해시를 계산하여 데이터 버전 식별자를 생성합니다."""
    info = f"{len(monthly)}_{monthly['date'].min()}_{monthly['date'].max()}"
    info += f"_{','.join(feature_cols)}"
    info += f"_{config['target_col']}"
    return hashlib.md5(info.encode()).hexdigest()[:8]


def save_best_model(trained_models, results_df, feature_cols,
                    monthly, config=None):
    """MAE 기준 최적 모델을 model/ 폴더에 저장하고,
    모든 모델의 메타정보를 model/history/에 기록합니다.
    """
    if config is None:
        config = CONFIG

    model_dir = config.get("model_dir", "./model")
    history_dir = config.get("history_dir", "./model/history")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_version = _compute_data_version(monthly, feature_cols, config)

    # --- 1) MAE 기준 최적 모델 저장 ---
    valid_results = results_df.dropna(subset=["MAE"])
    if len(valid_results) == 0:
        print("  ⚠ 유효한 모델 결과 없음 — 저장 건너뜀")
        return

    best_row = valid_results.sort_values("MAE", ascending=True).iloc[0]
    best_name = best_row["Model"]
    current_mae = float(best_row["MAE"])
    should_save = True
    prev_mae = None

    if best_name in trained_models:
        # --- 이전 best 모델과 성능 비교 ---
        meta_path = os.path.join(model_dir, "best_model_meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                prev_meta = json.load(f)
            prev_mae = prev_meta.get("metrics", {}).get("MAE")
            prev_name = prev_meta.get("model_name", "?")

            if prev_mae is not None:
                improvement = prev_mae - current_mae
                pct = (improvement / prev_mae * 100) if prev_mae != 0 else 0

                if current_mae < prev_mae:
                    print(f"\n  📈 성능 향상! "
                          f"이전({prev_name}) MAE={prev_mae:,.0f} → "
                          f"현재({best_name}) MAE={current_mae:,.0f} "
                          f"(↓{improvement:,.0f}, {pct:.1f}% 개선)")
                elif current_mae == prev_mae:
                    print(f"\n  ➡️ 성능 동일: MAE={current_mae:,.0f} — 모델 유지")
                    should_save = False
                else:
                    print(f"\n  📉 성능 하락: "
                          f"이전({prev_name}) MAE={prev_mae:,.0f} → "
                          f"현재({best_name}) MAE={current_mae:,.0f} "
                          f"(↑{-improvement:,.0f}, {-pct:.1f}% 악화)")
                    print(f"     ⚠ 이전 모델이 더 우수 — 갱신하지 않습니다.")
                    should_save = False

        if should_save:
            # 기존 best 모델 파일을 history로 백업
            existing_best = [f for f in os.listdir(model_dir)
                             if f.startswith("best_model") and f.endswith(".pkl")]
            for f in existing_best:
                src = os.path.join(model_dir, f)
                dst = os.path.join(history_dir, f"prev_{ts}_{f}")
                os.rename(src, dst)

            existing_meta = [f for f in os.listdir(model_dir)
                             if f.startswith("best_model") and f.endswith(".json")]
            for f in existing_meta:
                src = os.path.join(model_dir, f)
                dst = os.path.join(history_dir, f"prev_{ts}_{f}")
                os.rename(src, dst)

            # 새 best 모델 저장
            model_path = os.path.join(model_dir, "best_model.pkl")
            joblib.dump(trained_models[best_name], model_path)

            best_meta = {
                "model_name": best_name,
                "saved_at": ts,
                "data_version": data_version,
                "metrics": {
                    "MAE": float(best_row["MAE"]),
                    "MAPE(%)": float(best_row["MAPE(%)"]),
                    "MSE": float(best_row.get("MSE", 0)),
                    "RMSE": float(best_row["RMSE"]),
                },
                "features": feature_cols,
                "feature_count": len(feature_cols),
                "hyperparameters": config["model_params"].get(best_name, {}),
                "early_stopping": {
                    "patience": config.get("early_stopping_rounds", None),
                    "best_iteration": getattr(
                        trained_models[best_name], 'best_iteration',
                        getattr(trained_models[best_name],
                                'best_iteration_', None)),
                },
                "data_period": {
                    "start": str(monthly['date'].min()),
                    "end": str(monthly['date'].max()),
                    "months": len(monthly),
                },
                "target": config["target_col"],
                "forecast_start": config["forecast_start"],
                "forecast_months": config["forecast_months"],
            }
            meta_path = os.path.join(model_dir, "best_model_meta.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(best_meta, f, ensure_ascii=False, indent=2,
                          default=str)

            print(f"\n  🏆 최적 모델 저장: {model_path}")
            print(f"     모델: {best_name}, MAE={best_row['MAE']:,.0f}, "
                  f"MAPE={best_row['MAPE(%)']:.2f}%")
        else:
            print(f"\n  ℹ️ 기존 best 모델 유지 (MAE={prev_mae:,.0f})")

    # --- 2) 전체 모델 학습 이력 → history/ ---
    run_record = {
        "run_id": ts,
        "data_version": data_version,
        "target": config["target_col"],
        "forecast_start": config["forecast_start"],
        "forecast_months": config["forecast_months"],
        "valid_months": config.get("valid_months", 6),
        "early_stopping_rounds": config.get("early_stopping_rounds", None),
        "model_updated": should_save,
        "previous_best_mae": prev_mae,
        "data_period": {
            "start": str(monthly['date'].min()),
            "end": str(monthly['date'].max()),
            "months": len(monthly),
        },
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "lag_periods": config["lag_periods"],
        "rolling_windows": config["rolling_windows"],
        "best_model": best_name,
        "models": [],
    }

    for _, row in results_df.iterrows():
        model_name = row["Model"]
        # 조기 중단 정보 추출
        m = trained_models.get(model_name)
        best_iter = None
        if m is not None:
            best_iter = getattr(m, 'best_iteration',
                                getattr(m, 'best_iteration_', None))
        model_info = {
            "name": model_name,
            "metrics": {
                "MAE": _safe_float(row.get("MAE")),
                "MAPE(%)": _safe_float(row.get("MAPE(%)")),
                "MSE": _safe_float(row.get("MSE")),
                "RMSE": _safe_float(row.get("RMSE")),
            },
            "train_time_sec": _safe_float(row.get("Time(s)")),
            "hyperparameters": config["model_params"].get(model_name, {}),
            "best_iteration": best_iter,
            "is_best": model_name == best_name,
        }
        if "Error" in row and pd.notna(row.get("Error")):
            model_info["error"] = str(row["Error"])
        run_record["models"].append(model_info)

    # 앙상블 설정도 기록
    if config.get("ensemble"):
        run_record["ensemble_config"] = config["ensemble"]

    history_path = os.path.join(history_dir, f"run_{ts}.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2, default=str)
    print(f"  📋 학습 이력 저장: {history_path}")

    # --- 3) 이력 요약 CSV (누적) ---
    summary_path = os.path.join(history_dir, "history_summary.csv")
    summary_rows = []
    for _, row in results_df.iterrows():
        summary_rows.append({
            "run_id": ts,
            "data_version": data_version,
            "model": row["Model"],
            "MAE": _safe_float(row.get("MAE")),
            "MAPE(%)": _safe_float(row.get("MAPE(%)")),
            "MSE": _safe_float(row.get("MSE")),
            "RMSE": _safe_float(row.get("RMSE")),
            "train_time": _safe_float(row.get("Time(s)")),
            "is_best": row["Model"] == best_name,
            "features": len(feature_cols),
        })
    new_df = pd.DataFrame(summary_rows)

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path, encoding='utf-8-sig')
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  📈 누적 이력: {summary_path} ({len(combined)}건)")


def _safe_float(val):
    """NaN/None을 None으로 변환, 나머지는 float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def save_results(results_df, pred_dfs, config=None):
    """결과를 CSV/JSON으로 저장합니다."""
    if config is None:
        config = CONFIG
    if not config["save_results"]:
        return

    os.makedirs(config["results_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 모델 비교 결과
    csv_path = os.path.join(config["results_dir"],
                            f"model_comparison_{ts}.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 모델 비교: {csv_path}")

    # 전체 모델 예측 비교
    target = config["target_col"]
    all_preds = None
    for name, pdf in pred_dfs.items():
        cols = [config["date_col"], f'predicted_{target}']
        tmp = pdf[cols].rename(columns={f'predicted_{target}': name})
        if all_preds is None:
            all_preds = tmp
        else:
            all_preds = all_preds.merge(tmp, on=config["date_col"])
    if all_preds is not None:
        all_path = os.path.join(config["results_dir"],
                                f"forecast_all_{ts}.csv")
        all_preds.to_csv(all_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ 전체 예측 비교: {all_path}")

    # 설정 JSON
    json_path = os.path.join(config["results_dir"], f"config_{ts}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✓ 설정 저장: {json_path}")


# ============================================================
# MAIN: 전체 파이프라인 실행
# ============================================================

def main(config=None):
    """전체 ML 파이프라인을 실행합니다.

    Returns:
        results_df: 모델 비교 결과 DataFrame
        pred_dfs: 모델별 미래 예측 DataFrame dict
        trained_models: 학습된 모델 dict
    """
    if config is None:
        config = CONFIG

    print("=" * 60)
    print("  자금예측 ML — 앙상블 융합 모델")
    print("=" * 60)

    # 1) 데이터 로드
    print("\n[1/7] 데이터 로드")
    data = load_data(config)

    # 2) 월별 타겟 시계열
    print("\n[2/7] 월별 시계열 생성")
    monthly = prepare_monthly_target(data, config)

    # 3) 피처 엔지니어링
    print("\n[3/7] 피처 엔지니어링")
    df_model, feature_cols = create_features(monthly, config)

    # 4) 학습/검증 분할
    print("\n[4/7] 학습/검증 분할")
    X_train, y_train, X_valid, y_valid = split_timeseries(
        df_model, feature_cols, config)

    # 5) 모델 학습 & 평가
    print("\n[5/7] 모델 학습 & 평가")
    models = get_models(config)
    results, predictions, trained_models = train_all_models(
        models, X_train, y_train, X_valid, y_valid, config)

    # 결과표 (MAE 기준 오름차순 — 낮을수록 좋음)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE", ascending=True,
                                        na_position="last")

    print("\n" + "=" * 70)
    print("  📊 모델 비교 결과 (MAE 기준 정렬 — 낮을수록 좋음)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # 6) 최적 모델 저장 + 학습 이력
    print("\n[6/7] 모델 저장 & 이력 기록")
    save_best_model(trained_models, results_df, feature_cols,
                    monthly, config)

    # 7) 전체 데이터로 재학습 → 미래 예측
    print("\n[7/7] 미래 예측 (재귀적)")
    target = config["target_col"]
    pred_dfs = {}

    for name in models:
        try:
            fresh = get_models(config)
            if name not in fresh:
                continue
            model_full = fresh[name]
            model_full.fit(df_model[feature_cols], df_model[target])

            pred_df = recursive_forecast(
                model_full, monthly, feature_cols, config)
            pred_dfs[name] = pred_df

            target_key = f'predicted_{target}'
            print(f"\n  🔮 {name} 예측:")
            for _, row in pred_df.iterrows():
                print(f"     {row[config['date_col']]} : "
                      f"{row[target_key]:>20,.0f}")
        except Exception as e:
            print(f"  ❌ {name} 예측 실패: {e}")

    # 저장
    save_results(results_df, pred_dfs, config)

    # 시각화
    if config["plot_results"]:
        plot_model_comparison(results_df, monthly, pred_dfs,
                              y_valid, predictions, config)
        plot_feature_importance(trained_models, feature_cols, config)

    print("\n✅ 완료!")
    return results_df, pred_dfs, trained_models


# ============================================================
# 직접 실행
# ============================================================
if __name__ == "__main__":
    results_df, pred_dfs, trained_models = main()
