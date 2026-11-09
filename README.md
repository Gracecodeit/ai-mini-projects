# AI Mini Projects

멀티캠퍼스 AI 엔지니어 트랙 모듈 1 「[기본] AI 모델 실험과 검증」 미니프로젝트 모음입니다.

| 과제 | 주제 | 상태 |
|---|---|---|
| Day 1 (미니프로젝트 01) | 공유자전거 대여 수요 예측 (회귀) | 완료 |
| Day 2 (미니프로젝트 01) | 온라인 쇼핑 구매 여부 예측 (분류, PyTorch MLP) | 완료|

## Day 1. 공유자전거 대여량 예측 (회귀)

- **목표:** 시간 및 날씨 정보를 활용해 시간당 공유자전거 대여량(`cnt`) 예측
- **데이터:** UCI Bike Sharing Dataset (`hour.csv`, 17,379행 × 17열)
- **사용 기술:** Python 3.11, uv, pandas, NumPy, scikit-learn, matplotlib, joblib

### 프로젝트 자료

| 구분 | 파일 |
|---|---|
| 실험 코드 | [01_regression_bike_sharing.ipynb](mini-project-01/code/01_regression_bike_sharing.ipynb) |
| 실험 결과 | [모델 비교표](mini-project-01/results/01_model_comparison.csv), [최종 모델](mini-project-01/results/01_final_model.joblib) |
| 분석 | [데이터 분석](mini-project-01/docs/01_regression_data_analysis.md), [모델 분석](mini-project-01/docs/01_regression_model_analysis.md) |
| 핵심 의사결정 | [ADR](mini-project-01/docs/adr.md) |

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

| 모델 | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Dummy | 194.52 | 254.61 | -0.296 |
| Linear Regression | 99.97 | **124.36** | **0.691** |
| Random Forest (최종 후보 RF_B) | **96.63** | 144.08 | 0.585 |

최종 모델은 Validation MAE가 가장 낮았던 **RandomForestRegressor (n_estimators=100, max_depth=15)** 를 선택했습니다. Train과 Validation 데이터를 합쳐 다시 학습한 뒤 Test 데이터로 한 번 평가했습니다.

| Test 지표 | 결과 |
|---|---:|
| MAE | 66.17 |
| RMSE | 95.88 |
| R² | 0.80 |

MAE를 주요 모델 선택 기준으로 삼았지만, RMSE와 R²에서는 Linear Regression이 더 좋은 Validation 결과를 보였습니다. 따라서 최종 모델이 모든 평가 지표에서 가장 우수한 것은 아닙니다.

### 분석에서 확인한 사항과 한계

- `casual`과 `registered`의 합이 `cnt`와 같아 데이터 누수가 발생하므로 입력 특성에서 제외했습니다.
- 날짜 순서에 따라 데이터를 분할해 미래 데이터가 학습에 섞이지 않도록 했습니다.
- 새벽 3시와 출퇴근 시간대(8시, 17시)에서 예측 오차가 상대적으로 크게 나타났습니다.
- Random Forest의 평균 잔차는 -50.6으로, 실제값보다 높게 예측하는 경향이 관찰됐습니다.
- 시간대별 오차의 구체적인 원인은 추가 분석이 필요합니다.

### 폴더 구조

```
mini-project-01/
├── code/
│   └── 01_regression_bike_sharing.ipynb
├── data/
│   └── hour.csv
├── results/
│   ├── 01_model_comparison.csv
│   └── 01_final_model.joblib
└── docs/
    ├── 01_regression_data_analysis.md
    ├── 01_regression_model_analysis.md
    └── adr.md
```

### 실행 방법

```
git clone https://github.com/Gracecodeit/ai-mini-projects.git
cd ai-mini-projects
uv sync
```

VS Code에서 `mini-project-01/code/01_regression_bike_sharing.ipynb`를 열고 `.venv` 커널을 선택한 뒤 Restart 후 Run All을 실행합니다.

### 문서 관리 원칙

실험별 데이터 분석과 모델 분석은 번호가 붙은 문서에 기록하고, 주요 의사결정과 선택 근거는 `adr.md`에 정리합니다. 프로젝트의 핵심 결과와 실행 방법은 README에서 확인할 수 있도록 관리합니다.


## Day 2. 온라인 쇼핑 구매 여부 예측 (분류, PyTorch MLP)

- **목표:** 방문 세션의 행동 정보로 구매(`Revenue=1`) 여부 분류
- **데이터:** UCI Online Shoppers Purchasing Intention (`online_shoppers_intention.csv`, 12,330행 × 18열, 구매 15.47%)
- **사용 기술:** Python 3.11, uv, pandas, scikit-learn, PyTorch, joblib

### 프로젝트 자료

| 구분 | 파일 |
|---|---|
| 실험 코드 | [02_classification_online_shopping.ipynb](mini-project-01/code/02_classification_online_shopping.ipynb) |
| 실험 결과 | [모델 비교표](mini-project-01/results/02_model_comparison.csv), [임계값 비교표](mini-project-01/results/02_threshold_comparison.csv), [모델 설정](mini-project-01/results/02_model_config.json), [최종 모델](mini-project-01/results/02_final_classifier.joblib) |
| 분석 | [데이터 분석](mini-project-01/docs/02_classification_data_analysis.md), [모델 분석](mini-project-01/docs/02_classification_model_analysis.md) |
| 핵심 의사결정 | [ADR](mini-project-01/docs/adr.md) (의사결정 ③·④) |

### 실험 결과 (구매 클래스 기준, Validation)

| 모델 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| DummyClassifier | 0.845 | 0.000 | 0.000 | 0.000 |
| LogisticRegression | 0.882 | 0.734 | 0.367 | 0.490 |
| MLP-A (BCE) | 0.898 | 0.717 | 0.566 | 0.633 |
| **MLP-B (weighted BCE)** | 0.875 | 0.575 | 0.734 | **0.645** |

최종 모델은 Validation F1이 가장 높았던 **MLP-B, 임계값 0.5**이며, Train+Validation으로 17 epoch 재학습한 뒤 Test에서 한 번 평가했습니다.

| Test 지표 | 결과 |
|---|---:|
| Accuracy | 0.859 |
| Precision | 0.527 |
| Recall | 0.843 |
| F1 | 0.649 |

- 한계: 구매 고객 탐지율(Recall)은 높지만 FP가 216건 발생해 Precision이 낮습니다.
