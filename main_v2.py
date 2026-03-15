"""
=================================================================
자금예측 ML 고도화 파이프라인 (main_v2.py)
=================================================================
v1 대비 개선사항:
  1. 외생 변수 피처 (매출채권/매입채무 테이블 활용)
  2. 고급 시계열 피처 (추세, Fourier, EMA, 변화율)
  3. CatBoost, Prophet 모델 추가
  4. Optuna 기반 하이퍼파라미터 최적화
  5. 앙상블 가중치 최적화
  6. 피처 선택 (RFE / Importance 기반)
  7. 불확실성 추정 (Quantile Regression)

[사용법]
  A) 직접 실행:  python main_v2.py
  B) 최적화 실행: python main_v2.py --optimize
  C) 노트북 연동: from main_v2 import CONFIG_V2, main_v2
=================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
import time
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

warnings.filterwarnings('ignore')


# ============================================================
# [CONFIG_V2] 고도화 설정
# ============================================================

CONFIG_V2 = {
    # ---- 데이터 ----
    "data_dir": r"C:\Users\7slwm\Downloads\data_jeju\data",
    "data_files": {
        "acc_pay":  "acc_pay_df.parquet",   # 매입채무
        "acc_rec":  "acc_rec_df.parquet",   # 매출채권
        "ap_days":  "ap_days_df.parquet",   # 매입기일
        "ar_days":  "ar_days_df.parquet",   # 매출기일
        "coa":      "coa_df.parquet",       # 계정과목
        "purchase": "purchase_df.parquet",  # 매입
        "revenue":  "revenue_df.parquet",   # 매출
    },

    # ---- 타겟 변수 ----
    "target_col": "TOT_MN_MNAM",       # 총공급가액 (월별 합계)
    "main_table": "revenue",           # 메인 테이블

    # ---- 시계열 설정 ----
    "date_col": "DM_DATA",
    "window_size": 6,                  # 슬라이딩 윈도우 (소표본이므로 6개월 유지)
    "rolling_windows": [3, 6],         # 이동평균 윈도우
    "forecast_months": 6,
    "forecast_start": "2025-01",

    # ---- 방법론 설정 (고도화) ----
    "use_log_transform": True,
    "use_tscv": True,
    "tscv_n_splits": 4,               # TSCV fold (소표본이므로 4 유지)
    "tscv_valid_size": 6,
    
    # ---- 외생 피처 활성화 ----
    "use_exogenous_features": True,    # 매출채권/매입채무 피처 사용
    "use_advanced_features": True,     # 고급 시계열 피처 사용
    "use_fourier_features": False,     # Fourier 비활성 (소표본에 과적합 위험)
    "fourier_order": 2,               
    
    # ---- 피처 선택 ----
    "use_feature_selection": False,    # 소표본에서는 비활성화
    "feature_selection_method": "importance",
    "feature_selection_threshold": 0.005,

    # ---- 학습 설정 ----
    "valid_months": 6,
    "random_state": 42,
    "early_stopping_rounds": 30,       # 소표본이므로 낮춤

    # ---- 사용할 모델 ----
    "models": {
        # 기본 모델
        "XGBoost":            True,
        "LightGBM":           True,
        "CatBoost":           True,    # 신규 추가
        "GradientBoosting":   True,
        "RandomForest":       True,
        "Ridge":              True,
        # 시계열 특화 모델
        "Prophet":            False,    # Facebook Prophet (선택적)
        # 앙상블 융합 모델
        "Ensemble_Voting":    True,
        "Ensemble_Stacking":  True,
        "Ensemble_Weighted":  True,    # 신규: 최적화된 가중 앙상블
        # 비활성화
        "LinearRegression":   False,
        "Lasso":              False,
        "ElasticNet":         False,
    },

    # ---- 모델 하이퍼파라미터 (소표본 최적화) ----
    "model_params": {
        "XGBoost": {
            "n_estimators": 300, "max_depth": 3, "learning_rate": 0.1,
            "subsample": 0.7, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 3,
            "random_state": 42, "verbosity": 0,
        },
        "LightGBM": {
            "n_estimators": 300, "max_depth": 3, "learning_rate": 0.1,
            "num_leaves": 10, "min_child_samples": 5,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 3.0,
            "random_state": 42, "verbose": -1,
        },
        "CatBoost": {
            "iterations": 300, "depth": 3, "learning_rate": 0.1,
            "l2_leaf_reg": 5.0, "random_state": 42,
            "verbose": False, "allow_writing_files": False,
        },
        "GradientBoosting": {
            "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
            "min_samples_split": 5, "min_samples_leaf": 3,
            "subsample": 0.8, "random_state": 42,
        },
        "RandomForest": {
            "n_estimators": 100, "max_depth": 10,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "random_state": 42, "n_jobs": -1,
        },
        "Ridge": {"alpha": 1.0},
    },

    # ---- 앙상블 설정 (고도화) ----
    "ensemble": {
        "voting_weights": [2, 2, 2, 1],  # XGB, LGB, CatBoost, GBM
        "stacking_final": "Ridge",
        "stacking_cv": 3,                # 소표본이므로 3
        "optimize_weights": True,        # 앙상블 가중치 최적화
    },

    # ---- Optuna 최적화 설정 ----
    "optuna": {
        "enabled": False,               # 명령줄 --optimize로 활성화
        "n_trials": 30,                 # 탐색 횟수 (소표본이므로 축소)
        "timeout": 300,                 # 최대 시간(초)
        "models_to_optimize": ["XGBoost", "LightGBM", "CatBoost"],
    },

    # ---- 출력/저장 설정 ----
    "model_dir": "./model",
    "history_dir": "./model/history",
    "save_results": True,
    "results_dir": "./result",
    "plot_results": False,             # 비활성화 (터미널 실행)
    
    # ---- 실험 정보 ----
    "experiment_name": "고도화 v2: 외생피처 + CatBoost",
    "experiment_changes": [
        "외생 변수 피처 추가 (매출채권/매입채무 테이블)",
        "고급 시계열 피처 (추세, EMA)",
        "CatBoost 모델 추가",
        "앙상블 가중치 최적화",
        "소표본 최적화 하이퍼파라미터",
    ],
}


# ============================================================
# 컬럼명 한글 → 영어 매핑 (기존 유지)
# ============================================================

RENAME_MAP = {
    '기준년월': 'DW_BAS_NYYMM',
    '사업자등록번호': 'NO_BIZ',
    '법인등록번호': 'NO_CORPOR',
    '거래비중': 'TRAN_RATE',
    '매입처사업자번호': 'NO_BISOCIAL',
    '매입처명': 'NM_TRADE',
    '발생(지급)월': 'DM_DATA',
    '매입채무발생액': 'MN_AP',
    '매입채무지급액': 'MN_AP_PAY',
    '매입채무잔액': 'MN_AP_BAL',
    '총매입채무발생액': 'TOT_MN_AP',
    '총매입채무지급액': 'TOT_MN_AP_PAY',
    '총매입채무잔액': 'TOT_MN_AP_BAL',
    '매출처사업자번호': 'NO_BISOCIAL',
    '매출처명': 'NM_TRADE',
    '발생(회수)월': 'DM_DATA',
    '매출채권발생액': 'MN_AR',
    '매출채권회수액': 'MN_AR_RCV',
    '매출채권잔액': 'MN_AR_BAL',
    '총매출채권발생액': 'TOT_MN_AR',
    '총매출채권회수액': 'TOT_MN_AR_RCV',
    '총매출채권잔액': 'TOT_MN_AR_BAL',
    '기간구분': 'CD_TERM',
    '매입채무지급기간': 'AP_DAYS',
    '가중평균지급기간': 'AP_DAYS_W',
    '매출채권회수기간': 'AR_DAYS',
    '가중평균회수기간': 'AR_DAYS_W',
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
    # 추가 컬럼
    '지급기간': 'DPAYP',
    '지급율': 'RATE_PAYP',
    '회수기간': 'DPAYR',
    '회수율': 'RATE_PAYR',
}


# ============================================================
# 1. 데이터 로드 (기존 유지)
# ============================================================

def load_data(config: dict = None) -> Dict[str, pd.DataFrame]:
    """parquet 파일들을 로드합니다."""
    if config is None:
        config = CONFIG_V2
    data = {}
    for name, filename in config["data_files"].items():
        filepath = os.path.join(config["data_dir"], filename)
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            df = df.rename(columns=RENAME_MAP)
            data[name] = df
            print(f"  ✓ {name}: {df.shape}")
        else:
            print(f"  ✗ {name}: 파일 없음")
    return data


# ============================================================
# 2. 월별 타겟 시계열 생성 (기존 유지)
# ============================================================

def prepare_monthly_target(data: dict, config: dict = None) -> pd.DataFrame:
    """메인 테이블에서 월별 타겟을 집계합니다."""
    if config is None:
        config = CONFIG_V2
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
# 3. 외생 변수 피처 생성 (신규)
# ============================================================

def create_exogenous_features(data: dict, monthly: pd.DataFrame, 
                              config: dict = None) -> pd.DataFrame:
    """매출채권/매입채무 테이블에서 외생 변수를 생성합니다.
    
    추출 피처:
      - ar_recovery_rate: 매출채권 회수율
      - ar_recovery_days: 매출채권 회수기간 (일)
      - ap_payment_rate: 매입채무 지급율
      - ap_payment_days: 매입채무 지급기간 (일)
      - ar_total: 월별 매출채권 발생액 합계
      - ap_total: 월별 매입채무 발생액 합계
      - ar_ap_ratio: 매출채권/매입채무 비율
    """
    if config is None:
        config = CONFIG_V2
    
    date_col = config["date_col"]
    df = monthly.copy()
    
    print(f"\n  [외생 변수 피처 생성]")
    
    # 한글 컬럼명 매핑 (원본 데이터가 한글일 경우)
    AR_DAYS_MAP = {
        '사업자등록번호': 'NO_BIZ',
        '회수시작월': 'DM_ST',
        '회수종료월': 'DM_END',
        '매출처사업자번호': 'NO_BISOCIAL',
        '회수기간': 'DPAYR',
        '회수율': 'RATE_PAYR',
        '매출채권_가중평균_회수금액': 'MN_WCAR',
        '기간구분': 'CD_TERM',
        '매출채권_회수금액': 'MN_CAR',
    }
    
    AP_DAYS_MAP = {
        '사업자등록번호': 'NO_BIZ',
        '지급시작월': 'DM_ST',
        '지급종료월': 'DM_END',
        '기간구분': 'CD_TERM',
        '지급기간': 'DPAYP',
        '지급율': 'RATE_PAYP',
        '매입채무_가중평균_지급금액': 'MN_WPAP',
        '매입채무_지급금액': 'MN_PAP',
    }
    
    ACC_REC_MAP = {
        '사업자등록번호': 'NO_BIZ',
        '발생(회수)월': 'DM_DATA',
        '매출채권발생액': 'MN_AR',
    }
    
    ACC_PAY_MAP = {
        '사업자등록번호': 'NO_BIZ',
        '발생(지급)월': 'DM_DATA',
        '매입채무발생액': 'MN_AP',
    }
    
    # --- 매출채권 회수기간/회수율 (ar_days) ---
    if "ar_days" in data:
        ar_days = data["ar_days"].copy()
        ar_days = ar_days.rename(columns=AR_DAYS_MAP)
        
        # 최근 12개월 데이터만 사용 (CD_TERM='1')
        if 'CD_TERM' in ar_days.columns:
            ar_days = ar_days[ar_days['CD_TERM'] == '1']
        
        # 회수종료월 기준으로 집계 (DM_END)
        if 'DPAYR' in ar_days.columns and 'DM_END' in ar_days.columns:
            ar_agg = ar_days.groupby('DM_END').agg({
                'DPAYR': 'mean',      # 회수기간
                'RATE_PAYR': 'mean',  # 회수율
            }).reset_index()
            ar_agg.columns = ['DM_END', 'ar_recovery_days', 'ar_recovery_rate']
            ar_agg['DM_END'] = ar_agg['DM_END'].astype(str)
            
            # 월 포맷 맞추기 (YYYYMM -> YYYY-MM)
            ar_agg['join_key'] = ar_agg['DM_END'].apply(
                lambda x: f"{x[:4]}-{x[4:]}" if len(x) == 6 else x)
            
            df = df.merge(ar_agg[['join_key', 'ar_recovery_days', 'ar_recovery_rate']], 
                          left_on=date_col, right_on='join_key', how='left')
            df.drop('join_key', axis=1, inplace=True, errors='ignore')
            
            print(f"    ✓ 매출채권 회수기간/회수율 추가")
    
    # --- 매입채무 지급기간/지급율 (ap_days) ---
    if "ap_days" in data:
        ap_days = data["ap_days"].copy()
        ap_days = ap_days.rename(columns=AP_DAYS_MAP)
        
        if 'CD_TERM' in ap_days.columns:
            ap_days = ap_days[ap_days['CD_TERM'] == '1']
        
        # 지급종료월 기준으로 집계 (DM_END)
        if 'DPAYP' in ap_days.columns and 'DM_END' in ap_days.columns:
            ap_agg = ap_days.groupby('DM_END').agg({
                'DPAYP': 'mean',      # 지급기간
                'RATE_PAYP': 'mean',  # 지급율
            }).reset_index()
            ap_agg.columns = ['DM_END', 'ap_payment_days', 'ap_payment_rate']
            ap_agg['DM_END'] = ap_agg['DM_END'].astype(str)
            
            ap_agg['join_key'] = ap_agg['DM_END'].apply(
                lambda x: f"{x[:4]}-{x[4:]}" if len(x) == 6 else x)
            
            df = df.merge(ap_agg[['join_key', 'ap_payment_days', 'ap_payment_rate']], 
                          left_on=date_col, right_on='join_key', how='left')
            df.drop('join_key', axis=1, inplace=True, errors='ignore')
            
            print(f"    ✓ 매입채무 지급기간/지급율 추가")
    
    # --- 매출채권 발생액 (acc_rec) ---
    if "acc_rec" in data:
        acc_rec = data["acc_rec"].copy()
        acc_rec = acc_rec.rename(columns=ACC_REC_MAP)
        
        # DM_DATA 컬럼 찾기
        dm_col = 'DM_DATA' if 'DM_DATA' in acc_rec.columns else None
        mn_col = 'MN_AR' if 'MN_AR' in acc_rec.columns else None
        
        if dm_col and mn_col:
            ar_monthly = acc_rec.groupby(dm_col)[mn_col].sum().reset_index()
            ar_monthly.columns = ['dm_temp', 'ar_total']
            ar_monthly['dm_temp'] = ar_monthly['dm_temp'].astype(str)
            ar_monthly['join_key'] = ar_monthly['dm_temp'].apply(
                lambda x: f"{x[:4]}-{x[4:]}" if len(x) == 6 else x)
            
            df = df.merge(ar_monthly[['join_key', 'ar_total']], 
                          left_on=date_col, right_on='join_key', how='left')
            df.drop('join_key', axis=1, inplace=True, errors='ignore')
            print(f"    ✓ 매출채권 발생액 합계 추가")
    
    # --- 매입채무 발생액 (acc_pay) ---
    if "acc_pay" in data:
        acc_pay = data["acc_pay"].copy()
        acc_pay = acc_pay.rename(columns=ACC_PAY_MAP)
        
        dm_col = 'DM_DATA' if 'DM_DATA' in acc_pay.columns else None
        mn_col = 'MN_AP' if 'MN_AP' in acc_pay.columns else None
        
        if dm_col and mn_col:
            ap_monthly = acc_pay.groupby(dm_col)[mn_col].sum().reset_index()
            ap_monthly.columns = ['dm_temp', 'ap_total']
            ap_monthly['dm_temp'] = ap_monthly['dm_temp'].astype(str)
            ap_monthly['join_key'] = ap_monthly['dm_temp'].apply(
                lambda x: f"{x[:4]}-{x[4:]}" if len(x) == 6 else x)
            
            df = df.merge(ap_monthly[['join_key', 'ap_total']], 
                          left_on=date_col, right_on='join_key', how='left')
            df.drop('join_key', axis=1, inplace=True, errors='ignore')
            print(f"    ✓ 매입채무 발생액 합계 추가")
    
    # --- 매출채권/매입채무 비율 ---
    if 'ar_total' in df.columns and 'ap_total' in df.columns:
        df['ar_ap_ratio'] = df['ar_total'] / (df['ap_total'] + 1e-10)
        print(f"    ✓ 매출채권/매입채무 비율 추가")
    
    # --- 매입 테이블에서 추가 피처 ---
    if "purchase" in data:
        purchase = data["purchase"].copy()
        
        # 한글 컬럼명 처리
        PURCHASE_MAP = {
            '사업자등록번호': 'NO_BIZ',
            '공급받는시기': 'DM_DATA',  # purchase는 '공급받는시기' 사용
            '공급시기': 'DM_DATA',      # 둘 다 매핑
            '총공급가액': 'TOT_MN_MNAM',
        }
        purchase = purchase.rename(columns=PURCHASE_MAP)
        
        if 'TOT_MN_MNAM' in purchase.columns and 'DM_DATA' in purchase.columns:
            purch_monthly = (
                purchase.groupby(['NO_BIZ', 'DM_DATA'])['TOT_MN_MNAM']
                .first()
                .reset_index()
                .groupby('DM_DATA')['TOT_MN_MNAM']
                .sum()
                .reset_index()
            )
            purch_monthly.columns = ['dm_temp', 'purchase_total']
            purch_monthly['dm_temp'] = purch_monthly['dm_temp'].astype(str)
            
            # 월 포맷 맞추기
            purch_monthly['join_key'] = purch_monthly['dm_temp'].apply(
                lambda x: f"{x[:4]}-{x[4:]}" if len(x) == 6 and '-' not in x else x)
            
            df = df.merge(purch_monthly[['join_key', 'purchase_total']], 
                          left_on=date_col, right_on='join_key', how='left')
            df.drop('join_key', axis=1, inplace=True, errors='ignore')
            print(f"    ✓ 매입 총액 추가")
            
            # 매출/매입 비율
            target = config["target_col"]
            if target in df.columns and 'purchase_total' in df.columns:
                df['revenue_purchase_ratio'] = df[target] / (df['purchase_total'] + 1e-10)
                print(f"    ✓ 매출/매입 비율 추가")
    
    exog_cols = [c for c in df.columns if c not in monthly.columns or c == date_col]
    print(f"    → 외생 피처 {len(exog_cols) - 1}개 추가됨")
    
    return df


# ============================================================
# 4. 피처 엔지니어링 (고도화)
# ============================================================

def create_features_v2(monthly: pd.DataFrame, data: dict = None,
                       config: dict = None) -> Tuple[pd.DataFrame, List[str]]:
    """고급 시계열 피처를 생성합니다.
    
    생성 피처:
      - 슬라이딩 윈도우 래그 (val_t-1 ~ val_t-{window})
      - 월 계절성 (sin/cos 인코딩)
      - 이동평균/표준편차 (rolling_mean, rolling_std)
      - 지수이동평균 (EMA)
      - 변화율 (pct_change)
      - 추세 (선형 추세 계수)
      - Fourier 피처 (계절성)
      - 방향성 피처 (상승/하락 여부)
      - 외생 변수 (매출채권/매입채무 등)
    """
    if config is None:
        config = CONFIG_V2
    target = config["target_col"]
    window = config.get("window_size", 12)
    df = monthly.copy()
    
    # --- 외생 변수 피처 ---
    if config.get("use_exogenous_features", True) and data is not None:
        df = create_exogenous_features(data, df, config)
    
    print(f"\n  [기본 시계열 피처]")
    
    # --- 슬라이딩 윈도우 래그 ---
    for i in range(1, window + 1):
        df[f'val_t-{i}'] = df[target].shift(i)
    print(f"    ✓ 래그 피처: val_t-1 ~ val_t-{window}")
    
    # --- 월 계절성 (sin/cos 인코딩) ---
    df['month'] = df['date'].dt.month
    if config.get("use_advanced_features", True):
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        print(f"    ✓ 월 계절성 (sin/cos) 추가")
    
    # --- 이동평균/표준편차 ---
    for w in config.get("rolling_windows", [3, 6, 12]):
        df[f'rolling_mean_{w}'] = df[target].rolling(w).mean()
        df[f'rolling_std_{w}'] = df[target].rolling(w).std()
        df[f'rolling_min_{w}'] = df[target].rolling(w).min()
        df[f'rolling_max_{w}'] = df[target].rolling(w).max()
    print(f"    ✓ 이동 통계량 (mean, std, min, max) 추가")
    
    # --- 고급 피처 ---
    if config.get("use_advanced_features", True):
        # 지수이동평균 (EMA)
        for span in [3, 6, 12]:
            df[f'ema_{span}'] = df[target].ewm(span=span, adjust=False).mean()
        print(f"    ✓ 지수이동평균 (EMA) 추가")
        
        # 변화율
        df['pct_change_1'] = df[target].pct_change(1)
        df['pct_change_3'] = df[target].pct_change(3)
        df['pct_change_6'] = df[target].pct_change(6)
        print(f"    ✓ 변화율 (pct_change) 추가")
        
        # 추세 (최근 N개월 선형 회귀 기울기)
        def calc_trend(series, window=6):
            if len(series) < window:
                return np.nan
            trends = []
            for i in range(len(series)):
                if i < window - 1:
                    trends.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    if np.std(y) > 0:
                        slope = np.polyfit(x, y, 1)[0]
                    else:
                        slope = 0
                    trends.append(slope)
            return trends
        
        df['trend_6'] = calc_trend(df[target], 6)
        df['trend_12'] = calc_trend(df[target], 12)
        print(f"    ✓ 추세 피처 (선형 기울기) 추가")
        
        # 방향성 피처 (전월 대비 상승/하락)
        df['direction_1'] = (df[target].diff(1) > 0).astype(int)
        df['direction_3'] = (df[target].diff(3) > 0).astype(int)
        print(f"    ✓ 방향성 피처 추가")
        
        # 이동평균 대비 현재 값 비율
        rolling_windows = config.get("rolling_windows", [3, 6])
        if 6 in rolling_windows and f'rolling_mean_6' in df.columns:
            df['ratio_to_ma6'] = df[target] / (df[f'rolling_mean_6'] + 1e-10)
        if 12 in rolling_windows and f'rolling_mean_12' in df.columns:
            df['ratio_to_ma12'] = df[target] / (df[f'rolling_mean_12'] + 1e-10)
        print(f"    ✓ 이동평균 대비 비율 추가")
    
    # --- Fourier 피처 (계절성) ---
    if config.get("use_fourier_features", True):
        order = config.get("fourier_order", 3)
        # 연간 주기 (period=12)
        t = np.arange(len(df))
        for k in range(1, order + 1):
            df[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * t / 12)
            df[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * t / 12)
        print(f"    ✓ Fourier 피처 (order={order}) 추가")
    
    # --- 피처 컬럼 목록 ---
    exclude_cols = [config["date_col"], 'date', target]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # NaN 처리: 핵심 시계열 피처만 dropna, 나머지는 fillna
    core_features = [f'val_t-{i}' for i in range(1, config.get("window_size", 12) + 1)]
    core_features.extend(['month'])
    
    # 핵심 피처 NaN 행 제거
    df_model = df.dropna(subset=[c for c in core_features if c in df.columns]).copy()
    
    # 외생 피처 및 기타 피처 NaN을 0 또는 중앙값으로 채움
    for col in feature_cols:
        if col not in core_features and col in df_model.columns:
            if df_model[col].isna().any():
                # 수치형 피처: 중앙값으로 채움
                if df_model[col].dtype in ['float64', 'int64']:
                    median_val = df_model[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_model[col] = df_model[col].fillna(median_val)
                else:
                    df_model[col] = df_model[col].fillna(0)
    
    print(f"\n  ✓ 총 피처 수: {len(feature_cols)}개")
    print(f"  ✓ 학습 가능 데이터: {len(df_model)}건 "
          f"({df_model[config['date_col']].min()} ~ "
          f"{df_model[config['date_col']].max()})")
    
    return df_model, feature_cols


# ============================================================
# 5. 피처 선택 (신규)
# ============================================================

def select_features(X_train: pd.DataFrame, y_train: pd.Series,
                    feature_cols: List[str], config: dict = None) -> List[str]:
    """중요도 기반 피처 선택을 수행합니다."""
    if config is None:
        config = CONFIG_V2
    
    if not config.get("use_feature_selection", False):
        return feature_cols
    
    # 빈 데이터 체크
    if len(X_train) == 0 or len(feature_cols) == 0:
        return feature_cols
    
    method = config.get("feature_selection_method", "importance")
    threshold = config.get("feature_selection_threshold", 0.01)
    
    print(f"\n  [피처 선택: {method}]")
    
    if method == "importance":
        from lightgbm import LGBMRegressor
        
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"    ⚠ 피처 중요도 계산 실패: {e}")
            return feature_cols
        
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances / (importances.sum() + 1e-10)  # 정규화
        
        selected = importances[importances >= threshold].index.tolist()
        
        # 최소 10개 피처는 유지
        if len(selected) < 10:
            selected = importances.nlargest(min(len(feature_cols), 20)).index.tolist()
        
        dropped = [f for f in feature_cols if f not in selected]
        
        print(f"    ✓ 선택된 피처: {len(selected)}개 (threshold={threshold})")
        if dropped:
            print(f"    ✗ 제거된 피처: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
        
        return selected
    
    elif method == "rfe":
        from sklearn.feature_selection import RFE
        from lightgbm import LGBMRegressor
        
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        n_features = max(len(feature_cols) // 2, 10)
        
        try:
            rfe = RFE(model, n_features_to_select=n_features, step=1)
            rfe.fit(X_train, y_train)
            selected = [f for f, s in zip(feature_cols, rfe.support_) if s]
        except Exception as e:
            print(f"    ⚠ RFE 실패: {e}")
            return feature_cols
        
        print(f"    ✓ RFE 선택된 피처: {len(selected)}개")
        return selected
    
    return feature_cols


# ============================================================
# 6. 모델 정의 (CatBoost 추가)
# ============================================================

def get_models_v2(config: dict = None) -> Dict[str, object]:
    """활성화된 모델들을 반환합니다 (CatBoost 포함)."""
    if config is None:
        config = CONFIG_V2
    model_map = {}
    params = config["model_params"]

    # --- 기본 모델 ---
    if config["models"].get("Ridge"):
        from sklearn.linear_model import Ridge
        model_map["Ridge"] = Ridge(**params.get("Ridge", {}))

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
            model_map["LightGBM"] = LGBMRegressor(**params.get("LightGBM", {}))
        except ImportError:
            print("  ⚠ LightGBM 미설치 → pip install lightgbm")

    # --- CatBoost (신규) ---
    if config["models"].get("CatBoost"):
        try:
            from catboost import CatBoostRegressor
            model_map["CatBoost"] = CatBoostRegressor(**params.get("CatBoost", {}))
        except ImportError:
            print("  ⚠ CatBoost 미설치 → pip install catboost")

    # --- Prophet (선택적) ---
    if config["models"].get("Prophet"):
        try:
            from prophet import Prophet
            # Prophet은 별도 래퍼 클래스로 처리
            model_map["Prophet"] = "PROPHET_PLACEHOLDER"
        except ImportError:
            print("  ⚠ Prophet 미설치 → pip install prophet")

    # --- 앙상블 융합 모델 ---
    need_voting = config["models"].get("Ensemble_Voting")
    need_stacking = config["models"].get("Ensemble_Stacking")
    need_weighted = config["models"].get("Ensemble_Weighted")

    if need_voting or need_stacking or need_weighted:
        from sklearn.ensemble import VotingRegressor, StackingRegressor

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
        try:
            from catboost import CatBoostRegressor
            base_estimators.append(
                ('cat', CatBoostRegressor(**params.get("CatBoost", {}))))
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
                    estimators=base_estimators, weights=weights)

            if need_stacking:
                from sklearn.linear_model import Ridge
                model_map["Ensemble_Stacking"] = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=Ridge(alpha=1.0),
                    cv=ens_cfg.get("stacking_cv", 5))

    print(f"  ✓ 활성 모델 {len(model_map)}개: {list(model_map.keys())}")
    return model_map


# ============================================================
# 7. Optuna 하이퍼파라미터 최적화 (신규)
# ============================================================

def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                             X_valid: pd.DataFrame, y_valid: pd.Series,
                             config: dict = None) -> dict:
    """Optuna를 사용하여 하이퍼파라미터를 최적화합니다."""
    if config is None:
        config = CONFIG_V2
    
    optuna_cfg = config.get("optuna", {})
    if not optuna_cfg.get("enabled", False):
        return config["model_params"]
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠ Optuna 미설치 → pip install optuna")
        return config["model_params"]
    
    print("\n" + "=" * 60)
    print("  🔬 Optuna 하이퍼파라미터 최적화")
    print("=" * 60)
    
    n_trials = optuna_cfg.get("n_trials", 50)
    timeout = optuna_cfg.get("timeout", 600)
    models_to_optimize = optuna_cfg.get("models_to_optimize", 
                                        ["XGBoost", "LightGBM", "CatBoost"])
    
    optimized_params = config["model_params"].copy()
    
    # --- XGBoost 최적화 ---
    if "XGBoost" in models_to_optimize and config["models"].get("XGBoost"):
        print(f"\n  [XGBoost 최적화] {n_trials} trials...")
        
        def xgb_objective(trial):
            from xgboost import XGBRegressor
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42, 'verbosity': 0,
            }
            model = XGBRegressor(**params)
            model.set_params(early_stopping_rounds=30)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            pred = model.predict(X_valid)
            mae = np.mean(np.abs(y_valid.values - pred))
            return mae
        
        study = optuna.create_study(direction='minimize')
        study.optimize(xgb_objective, n_trials=n_trials, timeout=timeout, 
                       show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbosity'] = 0
        optimized_params['XGBoost'] = best_params
        print(f"    ✓ Best MAE: {study.best_value:,.0f}")
        print(f"    ✓ Best params: depth={best_params['max_depth']}, "
              f"lr={best_params['learning_rate']:.3f}")
    
    # --- LightGBM 최적화 ---
    if "LightGBM" in models_to_optimize and config["models"].get("LightGBM"):
        print(f"\n  [LightGBM 최적화] {n_trials} trials...")
        
        def lgb_objective(trial):
            from lightgbm import LGBMRegressor, early_stopping
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 5, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                'random_state': 42, 'verbose': -1,
            }
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      callbacks=[early_stopping(stopping_rounds=30, verbose=False)])
            pred = model.predict(X_valid)
            mae = np.mean(np.abs(y_valid.values - pred))
            return mae
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lgb_objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbose'] = -1
        optimized_params['LightGBM'] = best_params
        print(f"    ✓ Best MAE: {study.best_value:,.0f}")
        print(f"    ✓ Best params: depth={best_params['max_depth']}, "
              f"lr={best_params['learning_rate']:.3f}")
    
    # --- CatBoost 최적화 ---
    if "CatBoost" in models_to_optimize and config["models"].get("CatBoost"):
        print(f"\n  [CatBoost 최적화] {n_trials} trials...")
        
        def cat_objective(trial):
            from catboost import CatBoostRegressor
            params = {
                'iterations': trial.suggest_int('iterations', 100, 800),
                'depth': trial.suggest_int('depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
                'random_state': 42, 'verbose': False, 'allow_writing_files': False,
            }
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      early_stopping_rounds=30, verbose=False)
            pred = model.predict(X_valid)
            mae = np.mean(np.abs(y_valid.values - pred))
            return mae
        
        study = optuna.create_study(direction='minimize')
        study.optimize(cat_objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbose'] = False
        best_params['allow_writing_files'] = False
        optimized_params['CatBoost'] = best_params
        print(f"    ✓ Best MAE: {study.best_value:,.0f}")
        print(f"    ✓ Best params: depth={best_params['depth']}, "
              f"lr={best_params['learning_rate']:.3f}")
    
    return optimized_params


# ============================================================
# 8. 앙상블 가중치 최적화 (신규)
# ============================================================

def optimize_ensemble_weights(trained_models: dict, X_valid: pd.DataFrame,
                              y_valid: pd.Series, config: dict = None) -> dict:
    """앙상블 가중치를 최적화합니다."""
    if config is None:
        config = CONFIG_V2
    
    ens_cfg = config.get("ensemble", {})
    if not ens_cfg.get("optimize_weights", False):
        return {}
    
    print("\n  [앙상블 가중치 최적화]")
    
    # 단일 모델 예측값 수집
    base_models = ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting"]
    available_models = [m for m in base_models if m in trained_models]
    
    if len(available_models) < 2:
        print("    ⚠ 최적화에 필요한 모델 부족")
        return {}
    
    predictions = {}
    for name in available_models:
        try:
            pred = trained_models[name].predict(X_valid)
            predictions[name] = pred
        except Exception:
            pass
    
    if len(predictions) < 2:
        return {}
    
    # 그리드 서치로 최적 가중치 탐색
    from itertools import product
    
    best_mae = float('inf')
    best_weights = None
    
    weight_options = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    model_names = list(predictions.keys())
    
    for weights in product(weight_options, repeat=len(model_names)):
        if sum(weights) == 0:
            continue
        
        # 가중 평균 계산
        weighted_pred = np.zeros(len(y_valid))
        for w, name in zip(weights, model_names):
            weighted_pred += w * predictions[name]
        weighted_pred /= sum(weights)
        
        mae = np.mean(np.abs(y_valid.values - weighted_pred))
        
        if mae < best_mae:
            best_mae = mae
            best_weights = dict(zip(model_names, weights))
    
    print(f"    ✓ 최적 가중치: {best_weights}")
    print(f"    ✓ 가중 앙상블 MAE: {best_mae:,.0f}")
    
    return best_weights


# ============================================================
# 9. 학습/검증 분할 (기존 유지)
# ============================================================

def split_timeseries(df_model: pd.DataFrame, feature_cols: List[str],
                     config: dict = None) -> Tuple:
    """시계열 데이터를 학습/검증으로 분할합니다."""
    if config is None:
        config = CONFIG_V2
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
# 10. 평가 지표 (기존 유지 + 추가)
# ============================================================

def evaluate_model(y_true, y_pred, model_name: str = "") -> dict:
    """회귀 평가지표를 계산합니다."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.sum() > 0 else np.nan)
    
    # SMAPE (Symmetric MAPE)
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / 
                          (np.abs(y_true) + np.abs(y_pred) + 1e-10))

    return {
        "Model": model_name,
        "MAE": round(mae, 0),
        "MAPE(%)": round(mape, 2),
        "SMAPE(%)": round(smape, 2),
        "R2": round(r2, 4),
        "MSE": round(mse, 0),
        "RMSE": round(rmse, 0),
    }


# ============================================================
# 11. 모델 학습 (고도화)
# ============================================================

def _fit_model(name: str, model, X_train, y_train, X_valid, y_valid, 
               config: dict):
    """개별 모델 학습."""
    patience = config.get("early_stopping_rounds", 50)

    if name == "XGBoost":
        model.set_params(early_stopping_rounds=patience)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif name == "LightGBM":
        from lightgbm import early_stopping as lgb_early_stopping
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb_early_stopping(stopping_rounds=patience, verbose=False)])
    elif name == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                  early_stopping_rounds=patience, verbose=False)
    else:
        model.fit(X_train, y_train)
    
    return model


def train_all_models(models: dict, X_train, y_train, X_valid, y_valid,
                     config: dict = None) -> Tuple[list, dict, dict]:
    """모든 모델을 학습하고 평가합니다."""
    if config is None:
        config = CONFIG_V2
    results = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():
        if model == "PROPHET_PLACEHOLDER":
            continue  # Prophet은 별도 처리
        
        print(f"\n  🔄 {name} 학습 중...")
        try:
            start = time.time()
            model = _fit_model(name, model, X_train, y_train, X_valid, y_valid, config)
            elapsed = time.time() - start

            y_pred = model.predict(X_valid)
            metrics = evaluate_model(y_valid.values, y_pred, model_name=name)
            metrics["Time(s)"] = round(elapsed, 3)

            results.append(metrics)
            predictions[name] = y_pred
            trained_models[name] = model

            print(f"     ✅ MAE={metrics['MAE']:,.0f}, "
                  f"MAPE={metrics['MAPE(%)']}%, "
                  f"R2={metrics['R2']:.4f}")

        except Exception as e:
            print(f"     ❌ 실패: {e}")
            results.append({
                "Model": name, "MAE": None, "RMSE": None,
                "MAPE(%)": None, "R2": None, "MSE": None, "Error": str(e),
            })

    return results, predictions, trained_models


# ============================================================
# 12. TimeSeriesSplit 교차검증 (고도화)
# ============================================================

def tscv_evaluate(df_model: pd.DataFrame, feature_cols: List[str],
                  data: dict = None, config: dict = None) -> Tuple[pd.DataFrame, dict]:
    """TimeSeriesSplit 교차검증으로 모델을 평가합니다."""
    if config is None:
        config = CONFIG_V2

    n_splits = config.get("tscv_n_splits", 5)
    valid_size = config.get("tscv_valid_size", 6)
    target = config["target_col"]
    use_log = config.get("use_log_transform", False)

    df_sorted = df_model.sort_values('date').reset_index(drop=True)
    total = len(df_sorted)

    # Fold 생성
    folds = []
    for i in range(n_splits):
        valid_end = total - i * valid_size
        valid_start = valid_end - valid_size
        train_end = valid_start
        if train_end < 12:
            break
        folds.append({
            'fold': n_splits - i,
            'train': df_sorted.iloc[:train_end],
            'valid': df_sorted.iloc[valid_start:valid_end],
        })
    folds = sorted(folds, key=lambda x: x['fold'])

    print(f"\n  📋 TimeSeriesSplit ({len(folds)} folds, valid={valid_size}개월)")

    cv_detail = {}
    for fold in folds:
        X_tr = fold['train'][feature_cols]
        y_tr = fold['train'][target]
        X_vl = fold['valid'][feature_cols]
        y_vl = fold['valid'][target]

        models = get_models_v2(config)
        for name, model in models.items():
            if model == "PROPHET_PLACEHOLDER":
                continue
            try:
                model = _fit_model(name, model, X_tr, y_tr, X_vl, y_vl, config)
                y_pred = model.predict(X_vl)

                if use_log:
                    y_vl_eval = np.expm1(y_vl.values)
                    y_pred_eval = np.expm1(y_pred)
                else:
                    y_vl_eval = y_vl.values
                    y_pred_eval = y_pred

                metrics = evaluate_model(y_vl_eval, y_pred_eval, name)
                metrics['fold'] = fold['fold']

                if name not in cv_detail:
                    cv_detail[name] = []
                cv_detail[name].append(metrics)
            except Exception:
                pass

    # 평균 집계
    summary = []
    for name, fold_metrics in cv_detail.items():
        maes = [m['MAE'] for m in fold_metrics]
        mapes = [m['MAPE(%)'] for m in fold_metrics]
        rmses = [m['RMSE'] for m in fold_metrics]
        r2s = [m.get('R2', 0) for m in fold_metrics]
        summary.append({
            'Model': name,
            'MAE': round(np.mean(maes), 0),
            'MAPE(%)': round(np.mean(mapes), 2),
            'RMSE': round(np.mean(rmses), 0),
            'R2': round(np.mean(r2s), 4),
            'MAE_std': round(np.std(maes), 0),
            'Folds': len(fold_metrics),
        })

    summary_df = pd.DataFrame(summary).sort_values('MAE')
    
    print(f"\n  📊 TSCV 평균 결과:")
    for _, r in summary_df.iterrows():
        print(f"     {r['Model']:>20s}: MAE={r['MAE']:>15,.0f} "
              f"(±{r['MAE_std']:>12,.0f})  MAPE={r['MAPE(%)']:>6.2f}%")

    return summary_df, cv_detail


# ============================================================
# 13. 재귀적 미래 예측 (고도화)
# ============================================================

def recursive_forecast_v2(model, monthly: pd.DataFrame, feature_cols: List[str],
                          data: dict = None, config: dict = None) -> pd.DataFrame:
    """미래 N개월을 재귀적으로 예측합니다 (외생 변수 포함)."""
    if config is None:
        config = CONFIG_V2
    target = config["target_col"]
    window = config.get("window_size", 12)
    forecast_months = config["forecast_months"]
    forecast_start = config["forecast_start"]

    future_dates = pd.date_range(forecast_start, periods=forecast_months, freq='MS')
    history = monthly[['date', target]].copy()
    preds = []

    for tgt_date in future_dates:
        row = {}
        
        # 월 계절성
        row['month'] = tgt_date.month
        if config.get("use_advanced_features", True):
            row['month_sin'] = np.sin(2 * np.pi * tgt_date.month / 12)
            row['month_cos'] = np.cos(2 * np.pi * tgt_date.month / 12)

        # 슬라이딩 윈도우 래그
        for i in range(1, window + 1):
            lag_date = tgt_date - pd.DateOffset(months=i)
            match = history[history['date'] == lag_date]
            row[f'val_t-{i}'] = match[target].values[0] if len(match) > 0 else np.nan

        # 이동 통계량
        recent = history.sort_values('date')[target].values
        for w in config.get("rolling_windows", [3, 6, 12]):
            if len(recent) >= w:
                row[f'rolling_mean_{w}'] = np.mean(recent[-w:])
                row[f'rolling_std_{w}'] = np.std(recent[-w:])
                row[f'rolling_min_{w}'] = np.min(recent[-w:])
                row[f'rolling_max_{w}'] = np.max(recent[-w:])
            else:
                row[f'rolling_mean_{w}'] = np.nan
                row[f'rolling_std_{w}'] = np.nan
                row[f'rolling_min_{w}'] = np.nan
                row[f'rolling_max_{w}'] = np.nan

        # 고급 피처
        if config.get("use_advanced_features", True):
            for span in [3, 6, 12]:
                if len(recent) >= span:
                    alpha = 2 / (span + 1)
                    ema = recent[-1]
                    for v in recent[-span:-1][::-1]:
                        ema = alpha * v + (1 - alpha) * ema
                    row[f'ema_{span}'] = ema
                else:
                    row[f'ema_{span}'] = np.nan
            
            if len(recent) >= 2:
                row['pct_change_1'] = (recent[-1] - recent[-2]) / (recent[-2] + 1e-10)
            else:
                row['pct_change_1'] = np.nan
            if len(recent) >= 4:
                row['pct_change_3'] = (recent[-1] - recent[-4]) / (recent[-4] + 1e-10)
            else:
                row['pct_change_3'] = np.nan
            if len(recent) >= 7:
                row['pct_change_6'] = (recent[-1] - recent[-7]) / (recent[-7] + 1e-10)
            else:
                row['pct_change_6'] = np.nan
            
            # 추세
            if len(recent) >= 6:
                y = recent[-6:]
                x = np.arange(6)
                row['trend_6'] = np.polyfit(x, y, 1)[0] if np.std(y) > 0 else 0
            else:
                row['trend_6'] = np.nan
            if len(recent) >= 12:
                y = recent[-12:]
                x = np.arange(12)
                row['trend_12'] = np.polyfit(x, y, 1)[0] if np.std(y) > 0 else 0
            else:
                row['trend_12'] = np.nan
            
            # 방향성
            row['direction_1'] = 1 if len(recent) >= 2 and recent[-1] > recent[-2] else 0
            row['direction_3'] = 1 if len(recent) >= 4 and recent[-1] > recent[-4] else 0
            
            # 이동평균 대비 비율
            if len(recent) >= 6:
                row['ratio_to_ma6'] = recent[-1] / (np.mean(recent[-6:]) + 1e-10)
            else:
                row['ratio_to_ma6'] = np.nan
            if len(recent) >= 12:
                row['ratio_to_ma12'] = recent[-1] / (np.mean(recent[-12:]) + 1e-10)
            else:
                row['ratio_to_ma12'] = np.nan

        # Fourier 피처
        if config.get("use_fourier_features", True):
            order = config.get("fourier_order", 3)
            t = len(history)
            for k in range(1, order + 1):
                row[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * t / 12)
                row[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * t / 12)

        # 외생 변수 (마지막 값 유지 또는 0)
        for col in feature_cols:
            if col not in row:
                if col in monthly.columns:
                    last_val = monthly[col].dropna().iloc[-1] if len(monthly[col].dropna()) > 0 else 0
                    row[col] = last_val
                else:
                    row[col] = 0

        # 예측
        X_future = pd.DataFrame([row])[feature_cols]
        for col in X_future.columns:
            if X_future[col].isna().any():
                X_future[col] = X_future[col].fillna(0)
        
        pred = model.predict(X_future)[0]
        preds.append({
            'date': tgt_date,
            config["date_col"]: tgt_date.strftime('%Y-%m'),
            f'predicted_{target}': pred,
        })

        history = pd.concat(
            [history, pd.DataFrame([{'date': tgt_date, target: pred}])],
            ignore_index=True)

    return pd.DataFrame(preds)


# ============================================================
# 14. 최적 가중 앙상블 예측 (신규)
# ============================================================

def weighted_ensemble_predict(trained_models: dict, X: pd.DataFrame,
                              weights: dict) -> np.ndarray:
    """최적화된 가중치를 사용하여 앙상블 예측을 수행합니다."""
    predictions = []
    weight_list = []
    
    for name, w in weights.items():
        if name in trained_models and w > 0:
            pred = trained_models[name].predict(X)
            predictions.append(pred)
            weight_list.append(w)
    
    if not predictions:
        raise ValueError("No valid predictions")
    
    predictions = np.array(predictions)
    weight_list = np.array(weight_list)
    
    weighted_pred = np.average(predictions, axis=0, weights=weight_list)
    return weighted_pred


# ============================================================
# 15. 결과 저장 (기존 + 확장)
# ============================================================

def save_results(results_df: pd.DataFrame, pred_dfs: dict,
                 config: dict = None) -> dict:
    """결과를 저장합니다."""
    if config is None:
        config = CONFIG_V2
    if not config["save_results"]:
        return {}

    os.makedirs(config["results_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_paths = {}

    # 모델 비교 결과
    csv_path = os.path.join(config["results_dir"], f"model_comparison_{ts}.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    csv_paths["모델 비교 CSV"] = csv_path
    print(f"  ✓ 모델 비교: {csv_path}")

    # 전체 모델 예측
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
        all_path = os.path.join(config["results_dir"], f"forecast_all_{ts}.csv")
        all_preds.to_csv(all_path, index=False, encoding='utf-8-sig')
        csv_paths["예측 결과 CSV"] = all_path
        print(f"  ✓ 전체 예측 비교: {all_path}")

    # 설정 JSON
    json_path = os.path.join(config["results_dir"], f"config_v2_{ts}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    csv_paths["설정 JSON"] = json_path
    print(f"  ✓ 설정 저장: {json_path}")

    return csv_paths


# ============================================================
# MAIN: 전체 파이프라인 (고도화)
# ============================================================

def main_v2(config: dict = None, optimize: bool = False):
    """고도화된 ML 파이프라인을 실행합니다.
    
    Args:
        config: 설정 dict (None이면 CONFIG_V2 사용)
        optimize: Optuna 하이퍼파라미터 최적화 활성화 여부
    
    Returns:
        results_df, pred_dfs, trained_models
    """
    if config is None:
        config = CONFIG_V2.copy()
    
    if optimize:
        config["optuna"]["enabled"] = True

    use_log = config.get("use_log_transform", False)
    use_tscv = config.get("use_tscv", False)

    print("=" * 70)
    print("  🚀 자금예측 ML 고도화 파이프라인 (v2)")
    print("=" * 70)
    print(f"  - 로그 변환: {'ON' if use_log else 'OFF'}")
    print(f"  - TSCV: {'ON' if use_tscv else 'OFF'}")
    print(f"  - 외생 피처: {'ON' if config.get('use_exogenous_features') else 'OFF'}")
    print(f"  - Optuna 최적화: {'ON' if config['optuna']['enabled'] else 'OFF'}")

    # ── 1) 데이터 로드 ──
    print("\n[1/9] 데이터 로드")
    data = load_data(config)

    # ── 2) 월별 타겟 시계열 ──
    print("\n[2/9] 월별 시계열 생성")
    monthly = prepare_monthly_target(data, config)
    target = config["target_col"]

    # ── 3) 로그 변환 ──
    if use_log:
        print("\n[3/9] 로그 변환 적용")
        monthly_original = monthly.copy()
        neg_count = (monthly[target] <= 0).sum()
        if neg_count > 0:
            print(f"  ⚠ 음수/0 값 {neg_count}건 제거")
            monthly = monthly[monthly[target] > 0].copy()
        monthly[target] = np.log1p(monthly[target])
    else:
        print("\n[3/9] 로그 변환: 비활성")
        monthly_original = monthly.copy()

    # ── 4) 피처 엔지니어링 (고도화) ──
    print("\n[4/9] 피처 엔지니어링 (고도화)")
    df_model, feature_cols = create_features_v2(monthly, data, config)

    # ── 5) 학습/검증 분할 ──
    print("\n[5/9] 학습/검증 분할")
    X_train, y_train, X_valid, y_valid = split_timeseries(df_model, feature_cols, config)

    # ── 6) 피처 선택 ──
    if config.get("use_feature_selection", False):
        print("\n[6/9] 피처 선택")
        selected_features = select_features(X_train, y_train, feature_cols, config)
        X_train = X_train[selected_features]
        X_valid = X_valid[selected_features]
        feature_cols = selected_features
    else:
        print("\n[6/9] 피처 선택: 비활성")

    # ── 7) Optuna 하이퍼파라미터 최적화 ──
    if config["optuna"]["enabled"]:
        optimized_params = optimize_hyperparameters(X_train, y_train, X_valid, y_valid, config)
        config["model_params"] = optimized_params
    else:
        print("\n[7/9] Optuna 최적화: 비활성")

    # ── 8) TSCV 교차검증 ──
    tscv_summary = None
    if use_tscv:
        print("\n[8/9] TimeSeriesSplit 교차검증")
        tscv_summary, cv_detail = tscv_evaluate(df_model, feature_cols, data, config)
    else:
        print("\n[8/9] TimeSeriesSplit: 비활성")

    # ── 9) 모델 학습 ──
    print("\n[9/9] 모델 학습 & 평가")
    models = get_models_v2(config)
    results, predictions, trained_models = train_all_models(
        models, X_train, y_train, X_valid, y_valid, config)

    # 로그 역변환
    if use_log:
        print("\n  🔄 로그 역변환 → 원본 스케일 평가")
        y_valid_orig = np.expm1(y_valid.values)
        results_orig = []
        predictions_orig = {}
        for name, y_pred_log in predictions.items():
            y_pred_orig = np.expm1(y_pred_log)
            metrics = evaluate_model(y_valid_orig, y_pred_orig, model_name=name)
            for r in results:
                if r.get("Model") == name and r.get("Time(s)") is not None:
                    metrics["Time(s)"] = r["Time(s)"]
                    break
            results_orig.append(metrics)
            predictions_orig[name] = y_pred_orig
        results_df = pd.DataFrame(results_orig)
        y_valid_for_plot = pd.Series(y_valid_orig, index=y_valid.index)
        predictions_for_plot = predictions_orig
    else:
        results_df = pd.DataFrame(results)
        y_valid_for_plot = y_valid
        predictions_for_plot = predictions

    results_df = results_df.sort_values("MAE", ascending=True, na_position="last")

    # ── 앙상블 가중치 최적화 ──
    optimal_weights = optimize_ensemble_weights(trained_models, X_valid, y_valid, config)
    
    if optimal_weights and config["models"].get("Ensemble_Weighted"):
        print("\n  🔄 최적 가중 앙상블 평가...")
        try:
            weighted_pred = weighted_ensemble_predict(trained_models, X_valid, optimal_weights)
            if use_log:
                weighted_pred = np.expm1(weighted_pred)
            metrics = evaluate_model(y_valid_for_plot.values, weighted_pred, "Ensemble_Weighted")
            results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
            predictions_for_plot["Ensemble_Weighted"] = weighted_pred
            results_df = results_df.sort_values("MAE", ascending=True, na_position="last")
        except Exception as e:
            print(f"     ⚠ 가중 앙상블 실패: {e}")

    # 결과 출력
    print("\n" + "=" * 70)
    print("  📊 모델 비교 결과 (MAE 기준)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # ── 미래 예측 ──
    print("\n  🔮 미래 예측 (재귀적)")
    pred_dfs = {}
    best_model_name = results_df.iloc[0]["Model"]
    
    for name in models:
        if name not in trained_models:
            continue
        try:
            fresh = get_models_v2(config)
            if name not in fresh:
                continue
            model_full = fresh[name]
            if model_full == "PROPHET_PLACEHOLDER":
                continue
            model_full.fit(df_model[feature_cols], df_model[target])

            pred_df = recursive_forecast_v2(model_full, monthly, feature_cols, data, config)
            target_key = f'predicted_{target}'

            if use_log:
                pred_df[target_key] = np.expm1(pred_df[target_key])

            pred_dfs[name] = pred_df

            if name == best_model_name:
                print(f"\n  🏆 {name} (최적 모델) 예측:")
                for _, row in pred_df.iterrows():
                    print(f"     {row[config['date_col']]} : {row[target_key]:>20,.0f}")
        except Exception as e:
            print(f"  ❌ {name} 예측 실패: {e}")

    # 저장
    csv_paths = save_results(results_df, pred_dfs, config)

    # TSCV 저장
    if tscv_summary is not None and config["save_results"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tscv_path = os.path.join(config["results_dir"], f"tscv_summary_v2_{ts}.csv")
        tscv_summary.to_csv(tscv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ TSCV 결과: {tscv_path}")

    # 분석보고서 생성
    try:
        from report_utils import generate_experiment_report, find_latest_csv
        prev_csv_path = find_latest_csv(config["results_dir"], "model_comparison")
        prev_results_df = None
        if prev_csv_path:
            try:
                prev_results_df = pd.read_csv(prev_csv_path)
            except Exception:
                pass
        
        report_path = generate_experiment_report(
            results_df=results_df,
            config=config,
            experiment_name=config.get("experiment_name", "고도화 v2"),
            changes_summary=config.get("experiment_changes", []),
            csv_paths=csv_paths,
            previous_results_df=prev_results_df,
            previous_csv_path=prev_csv_path,
            feature_cols=feature_cols,
        )
        if report_path:
            print(f"  📎 분석보고서: {os.path.abspath(report_path)}")
    except Exception as e:
        print(f"  ⚠ 보고서 생성 실패: {e}")

    print("\n" + "=" * 70)
    print("  ✅ 고도화 파이프라인 완료!")
    print("=" * 70)
    
    best_row = results_df.iloc[0]
    print(f"\n  🏆 최적 모델: {best_row['Model']}")
    print(f"     MAE: {best_row['MAE']:,.0f}")
    print(f"     MAPE: {best_row['MAPE(%)']:.2f}%")
    print(f"     R2: {best_row.get('R2', 'N/A')}")

    return results_df, pred_dfs, trained_models


# ============================================================
# 직접 실행
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="자금예측 ML 고도화 파이프라인")
    parser.add_argument("--optimize", action="store_true", 
                        help="Optuna 하이퍼파라미터 최적화 활성화")
    args = parser.parse_args()
    
    results_df, pred_dfs, trained_models = main_v2(optimize=args.optimize)
