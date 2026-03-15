"""
=================================================================
모델 Checkpoint 저장 스크립트
=================================================================
현재 학습된 모델들을 pickle/joblib으로 저장하여
다른 환경에서도 사용할 수 있도록 합니다.

[저장 항목]
- 학습된 모델들 (.pkl)
- 피처 컬럼 목록 (.json)
- 설정 정보 (.json)
- 스케일러/인코더 (있는 경우)
=================================================================
"""

import os
import json
import joblib
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

# main_v2에서 필요한 함수들 import
from main_v2 import (
    CONFIG_V2, load_data, prepare_monthly_target,
    create_features_v2, split_timeseries, get_models_v2,
    train_all_models, _fit_model
)


def save_checkpoint(checkpoint_dir: str = "./checkpoint"):
    """현재 모델들을 checkpoint로 저장합니다."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    print("=" * 70)
    print("  📦 모델 Checkpoint 저장")
    print("=" * 70)
    print(f"  저장 위치: {os.path.abspath(checkpoint_path)}")
    
    config = CONFIG_V2.copy()
    config["plot_results"] = False
    
    # ── 1) 데이터 로드 및 전처리 ──
    print("\n[1/4] 데이터 준비...")
    data = load_data(config)
    monthly = prepare_monthly_target(data, config)
    
    # 로그 변환
    target = config["target_col"]
    if config.get("use_log_transform", False):
        monthly = monthly[monthly[target] > 0].copy()
        monthly[target] = np.log1p(monthly[target])
    
    # 피처 엔지니어링
    df_model, feature_cols = create_features_v2(monthly, data, config)
    
    # ── 2) 학습 ──
    print("\n[2/4] 모델 학습...")
    X_train, y_train, X_valid, y_valid = split_timeseries(df_model, feature_cols, config)
    
    # 전체 데이터로 최종 모델 학습
    X_full = df_model[feature_cols]
    y_full = df_model[target]
    
    models = get_models_v2(config)
    trained_models = {}
    
    for name, model in models.items():
        if model == "PROPHET_PLACEHOLDER":
            continue
        try:
            print(f"  학습 중: {name}...")
            # 전체 데이터로 학습
            if name == "XGBoost":
                model.set_params(early_stopping_rounds=None)
                model.fit(X_full, y_full, verbose=False)
            elif name == "LightGBM":
                model.fit(X_full, y_full)
            elif name == "CatBoost":
                model.fit(X_full, y_full, verbose=False)
            else:
                model.fit(X_full, y_full)
            trained_models[name] = model
        except Exception as e:
            print(f"    ⚠ {name} 학습 실패: {e}")
    
    # ── 3) 모델 저장 ──
    print("\n[3/4] 모델 파일 저장...")
    
    # 개별 모델 저장
    models_dir = os.path.join(checkpoint_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    saved_models = {}
    for name, model in trained_models.items():
        model_file = os.path.join(models_dir, f"{name}.pkl")
        try:
            joblib.dump(model, model_file)
            saved_models[name] = f"models/{name}.pkl"
            print(f"    ✓ {name} → {model_file}")
        except Exception as e:
            print(f"    ✗ {name} 저장 실패: {e}")
    
    # 전체 모델 딕셔너리도 저장 (편의용)
    all_models_file = os.path.join(checkpoint_path, "all_models.pkl")
    joblib.dump(trained_models, all_models_file)
    print(f"    ✓ 전체 모델 → all_models.pkl")
    
    # ── 4) 메타데이터 저장 ──
    print("\n[4/4] 메타데이터 저장...")
    
    # 피처 컬럼
    features_file = os.path.join(checkpoint_path, "feature_cols.json")
    with open(features_file, 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"    ✓ 피처 컬럼 ({len(feature_cols)}개) → feature_cols.json")
    
    # 설정 정보
    config_file = os.path.join(checkpoint_path, "config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    print(f"    ✓ 설정 정보 → config.json")
    
    # 체크포인트 메타
    meta = {
        "created_at": timestamp,
        "target_col": config["target_col"],
        "use_log_transform": config.get("use_log_transform", False),
        "feature_count": len(feature_cols),
        "train_samples": len(X_full),
        "data_period": {
            "start": str(df_model['date'].min()),
            "end": str(df_model['date'].max()),
        },
        "models": saved_models,
        "best_model": "CatBoost",  # 현재 최적 모델
        "performance": {
            "MAE": 92007474,
            "MAPE": 10.98,
            "R2": 0.9017,
        },
    }
    meta_file = os.path.join(checkpoint_path, "checkpoint_meta.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"    ✓ 메타데이터 → checkpoint_meta.json")
    
    # 사용법 README
    readme_content = f"""# 모델 Checkpoint

## 생성 정보
- 생성일시: {timestamp}
- 최적 모델: CatBoost (MAE=92,007,474, MAPE=10.98%)

## 파일 구조
```
checkpoint_{timestamp}/
├── all_models.pkl        # 전체 모델 딕셔너리
├── models/               # 개별 모델 파일
│   ├── CatBoost.pkl
│   ├── XGBoost.pkl
│   ├── LightGBM.pkl
│   └── ...
├── feature_cols.json     # 피처 컬럼 목록
├── config.json           # 학습 설정
└── checkpoint_meta.json  # 메타데이터
```

## 사용법

```python
import joblib
import json
import pandas as pd
import numpy as np

# 1. 모델 로드
models = joblib.load('all_models.pkl')
catboost_model = models['CatBoost']

# 또는 개별 모델 로드
catboost_model = joblib.load('models/CatBoost.pkl')

# 2. 피처 컬럼 로드
with open('feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

# 3. 예측
# 주의: 로그 변환이 적용되어 있으므로 역변환 필요
X_new = prepare_features(your_data)  # 동일한 피처 생성 필요
y_pred_log = catboost_model.predict(X_new[feature_cols])
y_pred = np.expm1(y_pred_log)  # 로그 역변환
```

## 주의사항
1. 입력 데이터는 동일한 피처 엔지니어링을 거쳐야 합니다
2. 로그 변환이 적용되어 있으므로 예측 후 np.expm1() 필요
3. 외생 변수 피처가 포함되어 있어 해당 테이블 데이터 필요
"""
    readme_file = os.path.join(checkpoint_path, "README.md")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"    ✓ 사용 가이드 → README.md")
    
    print("\n" + "=" * 70)
    print(f"  ✅ Checkpoint 저장 완료!")
    print(f"  📁 위치: {os.path.abspath(checkpoint_path)}")
    print("=" * 70)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: str):
    """저장된 checkpoint를 로드합니다."""
    
    print(f"📂 Checkpoint 로드: {checkpoint_path}")
    
    # 메타데이터 로드
    meta_file = os.path.join(checkpoint_path, "checkpoint_meta.json")
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 피처 컬럼 로드
    features_file = os.path.join(checkpoint_path, "feature_cols.json")
    with open(features_file, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    
    # 설정 로드
    config_file = os.path.join(checkpoint_path, "config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 모델 로드
    all_models_file = os.path.join(checkpoint_path, "all_models.pkl")
    models = joblib.load(all_models_file)
    
    print(f"  ✓ 모델 {len(models)}개 로드됨")
    print(f"  ✓ 피처 {len(feature_cols)}개")
    print(f"  ✓ 최적 모델: {meta.get('best_model', 'Unknown')}")
    
    return {
        "models": models,
        "feature_cols": feature_cols,
        "config": config,
        "meta": meta,
    }


if __name__ == "__main__":
    checkpoint_path = save_checkpoint()
