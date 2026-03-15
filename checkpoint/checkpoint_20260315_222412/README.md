# 모델 Checkpoint

## 생성 정보
- 생성일시: 20260315_222412
- 최적 모델: CatBoost (MAE=92,007,474, MAPE=10.98%)

## 파일 구조
```
checkpoint_20260315_222412/
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
