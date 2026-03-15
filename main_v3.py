"""
=================================================================
자금예측 ML 고도화 파이프라인 V3 (main_v3.py)
=================================================================
v2 대비 추가 개선사항:
  1. COA(계정과목) 테이블 피처 추가
  2. Box-Cox / Yeo-Johnson 타겟 변환
  3. Blending 앙상블 (Out-of-Fold predictions)
  4. 강화된 Optuna 최적화
  5. 교차 피처 (cross features)
  6. 자기상관 피처 (autocorrelation)

[사용법]
  A) 직접 실행:     python main_v3.py
  B) 최적화 실행:   python main_v3.py --optimize
  C) Box-Cox 사용:  python main_v3.py --boxcox
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
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# 기존 main_v2에서 필요한 컴포넌트 import
from main_v2 import (
    CONFIG_V2, RENAME_MAP,
    load_data, prepare_monthly_target,
    create_exogenous_features, create_features_v2,
    split_timeseries, _fit_model,
)


# ============================================================
# [CONFIG_V3] V3 추가 설정
# ============================================================

CONFIG_V3 = CONFIG_V2.copy()
CONFIG_V3.update({
    # ---- 타겟 변환 설정 ----
    "target_transform": "log",         # "log", "boxcox", "yeojohnson", "none"
    "boxcox_lambda": None,             # None이면 자동 추정
    
    # ---- COA 피처 설정 ----
    "use_coa_features": True,          # COA 테이블 피처 사용
    "coa_account_groups": {
        # 자산 계정 (111xxx)
        "current_asset":    [111100, 111150, 111151, 111157, 111180],  # 유동자산
        "cash_equivalent":  [111100, 111150],                          # 현금성자산
        "trade_receivable": [111300, 111400, 111410],                  # 매출채권
        # 부채 계정 (211xxx)
        "current_liability": [211100, 211200, 211300],                 # 유동부채
        "trade_payable":     [211100, 211200],                         # 매입채무
    },
    
    # ---- 고급 피처 설정 ----
    "use_cross_features": True,        # 교차 피처
    "use_autocorr_features": True,     # 자기상관 피처
    "autocorr_lags": [1, 3, 6],        # 자기상관 래그
    
    # ---- Blending 앙상블 ----
    "use_blending": True,              # Blending 앙상블 사용
    "blending_cv_folds": 3,            # OOF 생성용 CV fold
    
    # ---- 강화 Optuna 설정 ----
    "optuna": {
        "enabled": False,
        "n_trials": 50,                # 더 많은 탐색
        "timeout": 600,
        "models_to_optimize": ["XGBoost", "LightGBM", "CatBoost"],
        "prune": True,                 # 조기 종료
    },
    
    # ---- 실험 정보 ----
    "experiment_name": "고도화 V3: COA + Blending + BoxCox",
    "experiment_changes": [
        "COA(계정과목) 피처 추가",
        "Box-Cox/Yeo-Johnson 타겟 변환",
        "Blending 앙상블 (Out-of-Fold)",
        "강화된 Optuna 최적화",
        "교차 피처 & 자기상관 피처",
    ],
})


# ============================================================
# [COA 피처 생성]
# ============================================================

def create_coa_features(monthly: pd.DataFrame, coa_df: pd.DataFrame, 
                        config: dict) -> pd.DataFrame:
    """
    COA(계정과목) 테이블에서 월별 피처를 생성합니다.
    
    Parameters:
        monthly: 월별 집계 데이터 (date 컬럼 필요)
        coa_df: COA 테이블 (계정코드, 기준월, 총금액)
        config: 설정
    
    Returns:
        COA 피처가 추가된 DataFrame
    """
    if coa_df is None or coa_df.empty:
        print("    ⚠ COA 데이터가 없어 피처 생성 건너뜀")
        return monthly
    
    df = monthly.copy()
    
    # COA 컬럼명 정리
    coa = coa_df.copy()
    
    # 기준월을 datetime으로 변환
    if '기준월' in coa.columns:
        coa['date'] = pd.to_datetime(coa['기준월'], format='%Y-%m')
    else:
        print("    ⚠ COA 기준월 컬럼 없음")
        return monthly
    
    # 계정코드별 월별 합계
    if '계정코드' in coa.columns and '총금액' in coa.columns:
        coa_agg = coa.groupby(['date', '계정코드'])['총금액'].sum().reset_index()
    else:
        print("    ⚠ COA 필수 컬럼 없음")
        return monthly
    
    # 주요 계정 그룹별 합계 피처 생성
    account_groups = config.get("coa_account_groups", {})
    
    for group_name, account_codes in account_groups.items():
        # 해당 계정코드들 필터
        mask = coa_agg['계정코드'].isin(account_codes)
        if mask.sum() == 0:
            continue
            
        group_sum = coa_agg[mask].groupby('date')['총금액'].sum().reset_index()
        group_sum.columns = ['date', f'coa_{group_name}']
        
        df = df.merge(group_sum, on='date', how='left')
    
    # 추가 파생 피처 (금액이 큰 계정들의 비율)
    if 'coa_current_asset' in df.columns and 'coa_current_liability' in df.columns:
        # 유동비율 (current ratio)
        df['coa_current_ratio'] = (
            df['coa_current_asset'] / (df['coa_current_liability'] + 1)
        )
    
    if 'coa_trade_receivable' in df.columns and 'coa_trade_payable' in df.columns:
        # 매출채권/매입채무 비율
        df['coa_ar_ap_ratio'] = (
            df['coa_trade_receivable'] / (df['coa_trade_payable'] + 1)
        )
    
    # 결측값 처리
    coa_cols = [c for c in df.columns if c.startswith('coa_')]
    for col in coa_cols:
        df[col] = df[col].fillna(df[col].median())
    
    print(f"    ✓ COA 피처 {len(coa_cols)}개 추가됨")
    
    return df


# ============================================================
# [고급 피처 생성]
# ============================================================

def create_cross_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    주요 피처 간 교차 피처 생성
    """
    df = df.copy()
    new_features = []
    
    # 래그 피처와 외생 피처 간 교차
    lag_cols = [c for c in feature_cols if c.startswith('val_t-')]
    exog_cols = [c for c in feature_cols if c.startswith('ex_')]
    
    if lag_cols and exog_cols:
        # 가장 최근 래그와 주요 외생 피처 교차
        if 'val_t-1' in lag_cols:
            for ex_col in exog_cols[:3]:  # 상위 3개만
                cross_name = f'cross_{ex_col}'
                df[cross_name] = df['val_t-1'] * df.get(ex_col, 0)
                new_features.append(cross_name)
    
    # 계절성과 래그 교차
    if 'month_sin' in feature_cols and 'val_t-1' in feature_cols:
        df['cross_season_lag1'] = df['month_sin'] * df['val_t-1']
        new_features.append('cross_season_lag1')
    
    print(f"    ✓ 교차 피처 {len(new_features)}개 추가됨")
    return df, feature_cols + new_features


def create_autocorr_features(df: pd.DataFrame, target_col: str, 
                            lags: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """
    자기상관 피처 생성
    """
    df = df.copy()
    new_features = []
    
    if target_col not in df.columns:
        print(f"    ⚠ 타겟 컬럼 {target_col} 없음, 자기상관 피처 건너뜀")
        return df, new_features
    
    target = df[target_col].values
    
    for lag in lags:
        if len(target) > 2 * lag:
            autocorr_name = f'autocorr_lag{lag}'
            autocorr_values = []
            
            for i in range(len(target)):
                if i < 2 * lag:  # 충분한 데이터 필요
                    autocorr_values.append(np.nan)
                else:
                    arr1 = target[i-lag:i]
                    arr2 = target[i-2*lag:i-lag]
                    
                    if len(arr1) == len(arr2) and len(arr1) > 0:
                        try:
                            corr = np.corrcoef(arr1, arr2)[0, 1]
                            autocorr_values.append(corr if not np.isnan(corr) else 0)
                        except:
                            autocorr_values.append(0)
                    else:
                        autocorr_values.append(0)
            
            df[autocorr_name] = autocorr_values
            df[autocorr_name] = df[autocorr_name].fillna(0)
            new_features.append(autocorr_name)
    
    if new_features:
        print(f"    ✓ 자기상관 피처 {len(new_features)}개 추가됨")
    return df, new_features


# ============================================================
# [타겟 변환]
# ============================================================

class TargetTransformer:
    """타겟 변수 변환 클래스"""
    
    def __init__(self, method: str = "log", boxcox_lambda: float = None):
        self.method = method
        self.boxcox_lambda = boxcox_lambda
        self.fitted_lambda = None
        self.shift_value = 0
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """학습 데이터로 변환 파라미터 추정 후 변환"""
        y = np.array(y).astype(float)
        
        if self.method == "none":
            return y
        
        elif self.method == "log":
            return np.log1p(y)
        
        elif self.method == "boxcox":
            # Box-Cox는 양수만 가능
            if np.any(y <= 0):
                self.shift_value = abs(y.min()) + 1
                y = y + self.shift_value
            
            if self.boxcox_lambda is not None:
                self.fitted_lambda = self.boxcox_lambda
                y_transformed = stats.boxcox(y, lmbda=self.fitted_lambda)
            else:
                y_transformed, self.fitted_lambda = stats.boxcox(y)
            
            return y_transformed
        
        elif self.method == "yeojohnson":
            y_transformed, self.fitted_lambda = stats.yeojohnson(y)
            return y_transformed
        
        else:
            return y
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """학습된 파라미터로 변환 (테스트용)"""
        y = np.array(y).astype(float)
        
        if self.method == "none":
            return y
        elif self.method == "log":
            return np.log1p(y)
        elif self.method == "boxcox":
            y = y + self.shift_value
            return stats.boxcox(y, lmbda=self.fitted_lambda)
        elif self.method == "yeojohnson":
            return stats.yeojohnson(y, lmbda=self.fitted_lambda)
        else:
            return y
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """역변환"""
        y = np.array(y_transformed).astype(float)
        
        if self.method == "none":
            return y
        elif self.method == "log":
            return np.expm1(y)
        elif self.method == "boxcox":
            y_inv = self._inv_boxcox(y, self.fitted_lambda)
            return y_inv - self.shift_value
        elif self.method == "yeojohnson":
            return self._inv_yeojohnson(y, self.fitted_lambda)
        else:
            return y
    
    def _inv_boxcox(self, y, lmbda):
        """Inverse Box-Cox"""
        if lmbda == 0:
            return np.exp(y)
        else:
            return np.power(y * lmbda + 1, 1 / lmbda)
    
    def _inv_yeojohnson(self, y, lmbda):
        """Inverse Yeo-Johnson"""
        return stats.yeojohnson_normmax(y, brack=(-2, 2))  # 근사


# ============================================================
# [Blending 앙상블]
# ============================================================

def create_oof_predictions(X: pd.DataFrame, y: pd.Series, 
                           models: Dict, config: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Out-of-Fold 예측 생성 (Blending용)
    
    Parameters:
        X: 학습 피처
        y: 타겟
        models: 모델 딕셔너리
        config: 설정
    
    Returns:
        OOF 예측 DataFrame, 학습된 모델들
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    n_splits = config.get("blending_cv_folds", 3)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    oof_preds = pd.DataFrame(index=X.index)
    trained_models = {name: [] for name in models.keys()}
    
    base_models = ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting", "Ridge"]
    
    for name in base_models:
        if name not in models or models[name] == "PROPHET_PLACEHOLDER":
            continue
        
        oof_col = np.zeros(len(X))
        
        for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
            
            # 모델 복제 및 학습
            model = models[name].__class__(**models[name].get_params())
            
            try:
                if name == "XGBoost":
                    model.set_params(early_stopping_rounds=None)
                    model.fit(X_tr, y_tr, verbose=False)
                elif name == "CatBoost":
                    model.fit(X_tr, y_tr, verbose=False)
                else:
                    model.fit(X_tr, y_tr)
                
                oof_col[valid_idx] = model.predict(X_val)
                trained_models[name].append(model)
            except Exception as e:
                print(f"    ⚠ {name} fold {fold_idx} 학습 실패: {e}")
        
        oof_preds[f'oof_{name}'] = oof_col
    
    return oof_preds, trained_models


def optimize_blending_weights(oof_preds: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
    """
    OOF 예측 기반 최적 블렌딩 가중치 계산
    """
    from sklearn.metrics import mean_absolute_error
    
    oof_cols = [c for c in oof_preds.columns if c.startswith('oof_')]
    if len(oof_cols) < 2:
        return {}
    
    # 유효한 인덱스만 사용 (OOF가 0이 아닌 경우)
    valid_mask = oof_preds[oof_cols].sum(axis=1) != 0
    oof_valid = oof_preds.loc[valid_mask, oof_cols]
    y_valid = y_true.loc[valid_mask]
    
    if len(y_valid) < 5:
        return {}
    
    def objective(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))  # 정규화
        blend_pred = np.zeros(len(y_valid))
        for i, col in enumerate(oof_cols):
            blend_pred += weights[i] * oof_valid[col].values
        return mean_absolute_error(y_valid, blend_pred)
    
    # 초기 가중치
    n_models = len(oof_cols)
    init_weights = np.ones(n_models) / n_models
    
    # 최적화
    result = minimize(
        objective,
        init_weights,
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    
    # 결과
    opt_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    weights_dict = {col.replace('oof_', ''): w for col, w in zip(oof_cols, opt_weights)}
    
    return weights_dict


# ============================================================
# [V3 피처 엔지니어링 통합]
# ============================================================

def create_features_v3(monthly: pd.DataFrame, data: Dict[str, pd.DataFrame], 
                      config: dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    V3 피처 엔지니어링 (V2 + COA + 고급 피처)
    """
    target = config["target_col"]
    
    print("\n  [V3 피처 엔지니어링]")
    
    # ── 1) V2 기본 피처 (외생변수 + 시계열) ──
    # create_features_v2가 내부에서 create_exogenous_features도 호출함
    df, feature_cols = create_features_v2(monthly, data, config)
    
    # ── 2) COA 피처 (V3 신규) ──
    if config.get("use_coa_features", False) and 'coa' in data:
        df = create_coa_features(df, data['coa'], config)
        coa_cols = [c for c in df.columns if c.startswith('coa_')]
        feature_cols = feature_cols + [c for c in coa_cols if c not in feature_cols]
    
    # ── 3) 자기상관 피처 (V3 신규) ──
    if config.get("use_autocorr_features", False):
        autocorr_lags = config.get("autocorr_lags", [1, 3, 6])
        df, autocorr_features = create_autocorr_features(df, target, autocorr_lags)
        feature_cols = feature_cols + autocorr_features
    
    # ── 4) 교차 피처 (V3 신규) ──
    if config.get("use_cross_features", False):
        df, feature_cols = create_cross_features(df, feature_cols)
    
    # COA/외생 피처 결측값 채우기
    optional_cols = [c for c in feature_cols if c.startswith(('ex_', 'coa_', 'cross_', 'autocorr_'))]
    for col in optional_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # 최종 피처 목록 (실제 존재하는 컬럼만)
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # 최소 피처 수 보장
    if len(feature_cols) < 10:
        feature_cols = [c for c in df.columns 
                       if c not in [target, 'date', 'DW_BAS_NYYMM', 'NO_BIZ']][:20]
    
    print(f"\n  ✓ V3 총 피처 수: {len(feature_cols)}개")
    
    return df, feature_cols


# ============================================================
# [모델 생성 (V3)]
# ============================================================

def get_models_v3(config: dict) -> Dict:
    """V3 모델 생성 (V2 기반)"""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.ensemble import VotingRegressor, StackingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    
    params = config.get("model_params", {})
    model_flags = config.get("models", {})
    
    models = {}
    
    # 개별 모델
    if model_flags.get("XGBoost", True):
        models["XGBoost"] = XGBRegressor(**params.get("XGBoost", {}))
    
    if model_flags.get("LightGBM", True):
        models["LightGBM"] = LGBMRegressor(**params.get("LightGBM", {}))
    
    if model_flags.get("CatBoost", True):
        models["CatBoost"] = CatBoostRegressor(**params.get("CatBoost", {}))
    
    if model_flags.get("GradientBoosting", True):
        models["GradientBoosting"] = GradientBoostingRegressor(**params.get("GradientBoosting", {}))
    
    if model_flags.get("RandomForest", True):
        models["RandomForest"] = RandomForestRegressor(**params.get("RandomForest", {}))
    
    if model_flags.get("Ridge", True):
        models["Ridge"] = Ridge(**params.get("Ridge", {}))
    
    return models


# ============================================================
# [Optuna 최적화 (강화)]
# ============================================================

def optimize_with_optuna_v3(X_train, y_train, X_valid, y_valid, 
                           config: dict) -> Dict:
    """강화된 Optuna 하이퍼파라미터 최적화"""
    import optuna
    from sklearn.metrics import mean_absolute_error
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    best_params = {}
    
    # XGBoost 최적화
    def objective_xgb(trial):
        from xgboost import XGBRegressor
        
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "verbosity": 0,
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_valid)
        return mean_absolute_error(y_valid, pred)
    
    # CatBoost 최적화
    def objective_catboost(trial):
        from catboost import CatBoostRegressor
        
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "random_state": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_valid)
        return mean_absolute_error(y_valid, pred)
    
    # LightGBM 최적화
    def objective_lgbm(trial):
        from lightgbm import LGBMRegressor
        
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        return mean_absolute_error(y_valid, pred)
    
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 50)
    timeout = optuna_config.get("timeout", 600) // 3
    
    models_to_optimize = optuna_config.get("models_to_optimize", 
                                           ["XGBoost", "LightGBM", "CatBoost"])
    
    print("\n  [Optuna 하이퍼파라미터 최적화]")
    
    for model_name in models_to_optimize:
        print(f"    {model_name} 최적화 중...")
        
        if model_name == "XGBoost":
            study = optuna.create_study(direction="minimize")
            study.optimize(objective_xgb, n_trials=n_trials, timeout=timeout, 
                          show_progress_bar=False)
            best_params["XGBoost"] = study.best_params
            print(f"      → Best MAE: {study.best_value:.2f}")
            
        elif model_name == "CatBoost":
            study = optuna.create_study(direction="minimize")
            study.optimize(objective_catboost, n_trials=n_trials, timeout=timeout,
                          show_progress_bar=False)
            best_params["CatBoost"] = study.best_params
            print(f"      → Best MAE: {study.best_value:.2f}")
            
        elif model_name == "LightGBM":
            study = optuna.create_study(direction="minimize")
            study.optimize(objective_lgbm, n_trials=n_trials, timeout=timeout,
                          show_progress_bar=False)
            best_params["LightGBM"] = study.best_params
            print(f"      → Best MAE: {study.best_value:.2f}")
    
    return best_params


# ============================================================
# [V3 학습 및 평가 파이프라인]
# ============================================================

def train_and_evaluate_v3(df_model: pd.DataFrame, feature_cols: List[str],
                         config: dict, transformer: TargetTransformer) -> Dict:
    """V3 학습 및 평가 파이프라인"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    target = config["target_col"]
    
    # 학습/검증 분할
    X_train, y_train, X_valid, y_valid = split_timeseries(df_model, feature_cols, config)
    
    results = {}
    
    # ── Optuna 최적화 (옵션) ──
    optuna_params = {}
    if config.get("optuna", {}).get("enabled", False):
        optuna_params = optimize_with_optuna_v3(X_train, y_train, X_valid, y_valid, config)
    
    # 모델 생성 (Optuna 파라미터 적용)
    models = get_models_v3(config)
    
    if optuna_params:
        for name, params in optuna_params.items():
            if name in models:
                models[name].set_params(**params)
    
    # ── Blending 앙상블 (옵션) ──
    blending_weights = {}
    if config.get("use_blending", False):
        print("\n  [Blending 앙상블]")
        X_full = df_model[feature_cols]
        y_full = df_model[target]
        
        oof_preds, trained_models_oof = create_oof_predictions(X_full, y_full, models, config)
        blending_weights = optimize_blending_weights(oof_preds, y_full)
        
        if blending_weights:
            print(f"    ✓ 최적 가중치: {blending_weights}")
    
    # ── 모델 학습 및 평가 ──
    print("\n  [모델 학습 및 평가]")
    
    for name, model in models.items():
        try:
            # 학습 (_fit_model은 model 하나만 반환)
            model = _fit_model(name, model, X_train, y_train, X_valid, y_valid, config)
            
            # 예측
            pred_valid_transformed = model.predict(X_valid)
            
            # 역변환
            pred_valid = transformer.inverse_transform(pred_valid_transformed)
            y_valid_orig = transformer.inverse_transform(y_valid.values)
            
            # 음수 예측 처리
            pred_valid = np.maximum(pred_valid, 0)
            
            # 평가
            mae = mean_absolute_error(y_valid_orig, pred_valid)
            rmse = np.sqrt(mean_squared_error(y_valid_orig, pred_valid))
            mape = np.mean(np.abs((y_valid_orig - pred_valid) / (y_valid_orig + 1e-8))) * 100
            r2 = r2_score(y_valid_orig, pred_valid)
            
            results[name] = {
                "model": model,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
            }
            
            print(f"    {name:20s}: MAE={mae:,.0f}, MAPE={mape:.2f}%, R²={r2:.4f}")
            
        except Exception as e:
            print(f"    {name}: 실패 - {e}")
    
    # ── Blending 예측 ──
    if blending_weights:
        print("\n  [Blending 앙상블 예측]")
        blend_pred = np.zeros(len(X_valid))
        total_weight = sum(blending_weights.values())
        
        for name, weight in blending_weights.items():
            if name in results:
                model = results[name]["model"]
                pred = model.predict(X_valid)
                blend_pred += (weight / total_weight) * pred
        
        # 역변환
        blend_pred_orig = transformer.inverse_transform(blend_pred)
        blend_pred_orig = np.maximum(blend_pred_orig, 0)
        y_valid_orig = transformer.inverse_transform(y_valid.values)
        
        mae = mean_absolute_error(y_valid_orig, blend_pred_orig)
        mape = np.mean(np.abs((y_valid_orig - blend_pred_orig) / (y_valid_orig + 1e-8))) * 100
        r2 = r2_score(y_valid_orig, blend_pred_orig)
        
        results["Blending"] = {
            "model": None,
            "weights": blending_weights,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
        }
        print(f"    Blending: MAE={mae:,.0f}, MAPE={mape:.2f}%, R²={r2:.4f}")
    
    return results


# ============================================================
# [메인 함수]
# ============================================================

def main_v3(config: dict = None) -> Dict:
    """V3 메인 실행 함수"""
    
    if config is None:
        config = CONFIG_V3
    
    print("=" * 70)
    print("  🚀 자금예측 ML 파이프라인 V3")
    print("=" * 70)
    print(f"  실험: {config.get('experiment_name', '고도화 V3')}")
    print(f"  타겟 변환: {config.get('target_transform', 'log')}")
    
    # ── 1) 데이터 로드 ──
    print("\n[1/4] 데이터 로드...")
    data = load_data(config)
    monthly = prepare_monthly_target(data, config)
    
    # ── 2) 타겟 변환 ──
    print("\n[2/4] 타겟 변환...")
    target = config["target_col"]
    
    transform_method = config.get("target_transform", "log")
    transformer = TargetTransformer(method=transform_method)
    
    # 양수 데이터만 사용
    monthly = monthly[monthly[target] > 0].copy()
    monthly[target] = transformer.fit_transform(monthly[target].values)
    
    print(f"  ✓ 변환 방법: {transform_method}")
    if transform_method == "boxcox":
        print(f"  ✓ Box-Cox lambda: {transformer.fitted_lambda:.4f}")
    
    # ── 3) 피처 엔지니어링 ──
    print("\n[3/4] 피처 엔지니어링...")
    df_model, feature_cols = create_features_v3(monthly, data, config)
    
    # ── 4) 학습 및 평가 ──
    print("\n[4/4] 모델 학습 및 평가...")
    results = train_and_evaluate_v3(df_model, feature_cols, config, transformer)
    
    # ── 결과 요약 ──
    print("\n" + "=" * 70)
    print("  📊 최종 결과 요약")
    print("=" * 70)
    
    # 최적 모델 선정
    best_model = min(results.keys(), key=lambda k: results[k].get("MAPE", float('inf')))
    best_result = results[best_model]
    
    print(f"\n  🏆 Best Model: {best_model}")
    print(f"     MAE:  {best_result['MAE']:,.0f}")
    print(f"     MAPE: {best_result['MAPE']:.2f}%")
    print(f"     R²:   {best_result['R2']:.4f}")
    
    # 결과 저장
    if config.get("save_results", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = config.get("results_dir", "./result")
        os.makedirs(results_dir, exist_ok=True)
        
        # 결과 CSV
        results_data = []
        for name, res in results.items():
            results_data.append({
                "model": name,
                "MAE": res.get("MAE"),
                "MAPE": res.get("MAPE"),
                "R2": res.get("R2"),
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(
            os.path.join(results_dir, f"v3_comparison_{timestamp}.csv"),
            index=False, encoding='utf-8-sig'
        )
        
        print(f"\n  💾 결과 저장: {results_dir}/v3_comparison_{timestamp}.csv")
    
    return results


# ============================================================
# [스크립트 실행]
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="자금예측 ML V3")
    parser.add_argument("--optimize", action="store_true", help="Optuna 최적화 활성화")
    parser.add_argument("--boxcox", action="store_true", help="Box-Cox 변환 사용")
    parser.add_argument("--yeojohnson", action="store_true", help="Yeo-Johnson 변환 사용")
    parser.add_argument("--no-blending", action="store_true", help="Blending 비활성화")
    
    args = parser.parse_args()
    
    config = CONFIG_V3.copy()
    
    if args.optimize:
        config["optuna"]["enabled"] = True
        print("✓ Optuna 최적화 활성화")
    
    if args.boxcox:
        config["target_transform"] = "boxcox"
        print("✓ Box-Cox 변환 사용")
    
    if args.yeojohnson:
        config["target_transform"] = "yeojohnson"
        print("✓ Yeo-Johnson 변환 사용")
    
    if args.no_blending:
        config["use_blending"] = False
        print("✓ Blending 비활성화")
    
    main_v3(config)
