# AI Mini Projects

멀티캠퍼스 모듈 1 ML/DL 기초 미니프로젝트입니다.

## Day 1. 공유자전거 대여량 예측 (회귀)

- **목표:** 시간 및 날씨 정보를 활용해 시간당 공유자전거 대여량(`cnt`) 예측
- **데이터:** UCI Bike Sharing Dataset (`hour.csv`)
- **사용 기술:** Python 3.11, pandas, NumPy, scikit-learn, matplotlib, joblib

### 진행 과정

1. 데이터 탐색 및 기초 통계 확인
2. 날짜 기준 Train / Validation / Test 분할
3. 데이터 누수 확인 및 입력 특성 선정
4. 특성공학 및 전처리
5. Dummy 및 Linear Regression 기준 모델 비교
6. Decision Tree와 Random Forest 실험
7. 잔차 및 시간대별 예측 오류 분석
8. 최종 모델 재학습, Test 평가 및 저장

### 실험 결과

Validation 기준 주요 모델 성능은 다음과 같습니다.

| 모델 | Validation MAE |
|---|---:|
| Dummy | 194.52 |
| Linear Regression | 99.97 |
| Random Forest (최종 후보 RF_B) | 96.63 |

최종 모델은 Validation MAE가 가장 낮았던 **RandomForestRegressor (n_estimators=100, max_depth=15)** 를 선택했습니다.

- **Test MAE:** 66.17
- **Test RMSE:** 95.88
- **Test R²:** 0.80

MAE를 주요 모델 선택 기준으로 삼았지만, RMSE와 R²에서는 Linear Regression이 더 좋은 Validation 결과를 보였습니다. 따라서 최종 모델이 모든 평가 지표에서 가장 우수한 것은 아닙니다.

### 분석에서 확인한 사항과 한계

- `casual`과 `registered`의 합이 `cnt`와 같아 데이터 누수 가능성이 있으므로 입력 특성에서 제외했습니다.
- 날짜 순서에 따라 데이터를 분할해 미래 데이터가 학습에 섞이지 않도록 했습니다.
- 출퇴근 시간대와 일부 새벽 시간대에서 예측 오차가 상대적으로 크게 나타났습니다.
- Random Forest의 평균 잔차는 -50.6으로, 실제값보다 높게 예측하는 경향이 관찰됐습니다.
- 시간대별 오차의 구체적인 원인은 추가 분석이 필요합니다.

### 폴더 구조

```text
mini-project-01/
├── code/
│   └── 01_regression_bike_demand.ipynb
├── data/
│   └── hour.csv
├── results/
│   └── 01_final_model.joblib
└── docs/
    ├── 01_regression_data_analysis.md
    ├── 01_regression_model_analysis.md
    └── adr.md
```

### 실행 방법

```bash
git clone https://github.com/<내아이디>/ai-mini-projects.git
cd ai-mini-projects
uv sync
```

VS Code에서 `mini-project-01/code/01_regression_bike_demand.ipynb`를 열고 `.venv` 커널을 선택한 뒤 실행합니다.

### 분석 문서

- [데이터 분석](mini-project-01/docs/01_regression_data_analysis.md)
- [실험 분석](mini-project-01/docs/01_regression_model_analysis.md)
- [의사결정 기록(ADR)](mini-project-01/docs/adr.md)

### 문서 관리 원칙

실험별 데이터 분석과 모델 분석은 번호가 붙은 문서에 기록하고, 주요 의사결정과 선택 근거는 `adr.md`에 정리합니다. 프로젝트의 핵심 결과와 실행 방법은 README에서 확인할 수 있도록 관리합니다.
