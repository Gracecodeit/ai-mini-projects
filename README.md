<<<<<<< HEAD
# AI Mini Projects
멀티캠퍼스 AI 엔지니어 트랙 · 모듈 1 「[기본] AI 모델 실험과 검증」 미니프로젝트 모음입니다.
모델을 만드는 것보다 **여러 모델을 비교하고, 검증하고, 근거를 남겨 선택하는 과정**에 집중했습니다.
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
## 한눈에 보기
| Day | 주제 | 유형 | 최종 모델 | Test 결과 |
|---|---|---|---|---|
| 1 | 공유자전거 시간당 대여량 예측 | 회귀 | RandomForestRegressor (100그루, 깊이 15) | MAE 66.17 / RMSE 95.88 / R² 0.80 |
| 2 | 온라인 쇼핑 구매 여부 예측 | 이진 분류 | PyTorch MLP-B (weighted BCE, 임계값 0.5) | Recall 0.843 / F1 0.649 |
### 실험 원칙
- **데이터 누수 차단**: 정답을 그대로 만드는 컬럼(`casual + registered = cnt`)은 입력에서 제외
- **올바른 분할**: Day 1은 날짜순 분할(미래 정보 차단), Day 2는 `stratify` 계층화 분할(구매 비율 유지)
- **Train에만 fit**: 스케일러·인코더 기준은 Train에서만 학습하고 Validation/Test에는 transform만 적용
- **Baseline부터 비교**: Dummy → Linear/Logistic → Tree/RF/MLP 순으로 성능 향상 여부 확인
- **Test는 한 번만**: 모델·임계값은 Validation으로 결정하고 Test는 최종 확인에만 사용
- **의사결정 기록**: 지표·모델·임계값 선택 근거를 [ADR](mini-project-01/docs/adr.md)에 정리
## 폴더 구조
```
ai-mini-projects/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
└── mini-project-01/
├── code/
│ ├── 01_regression_bike_sharing.ipynb
│ └── 02_classification_online_shopping.ipynb
├── data/
│ ├── hour.csv
│ └── online_shoppers_intention.csv
├── results/
│ ├── 01_model_comparison.csv
│ ├── 01_final_model.joblib
│ ├── 02_model_comparison.csv
│ ├── 02_threshold_comparison.csv
│ ├── 02_model_config.json
AI Mini Projects 가이드 · 3 / 70
│ └── 02_final_classifier.joblib
└── docs/
├── 01_regression_data_analysis.md
├── 01_regression_model_analysis.md
├── 02_classification_data_analysis.md
├── 02_classification_model_analysis.md
└── adr.md
```
## Day 1. 공유자전거 대여량 예측 (회귀)
- **데이터**: UCI Bike Sharing `hour.csv` (17,379행 × 17열, 결측·중복 0)
- **분할**: Train 12,139 (~2012-05-25) / Validation 2,640 (~2012-09-12) / Test 2,600 (~2012-12-31)
- **특성공학**: 근무일 07–09시·17–19시를 표시하는 `is_commute_hour`
- **전처리**: 범주형 OneHot, 수치형 StandardScaler, 이진형 그대로 → 60개 특성
- **주 지표**: MAE
| 모델 | Train MAE | Valid MAE | Valid RMSE | Valid R² |
|---|---:|---:|---:|---:|
| **RF_B (100, 깊이 15)** | 22.91 | **96.63** | 144.08 | 0.585 |
| Linear | 55.33 | 99.97 | **124.36** | **0.691** |
| RF_A (50, 깊이 10) | 38.66 | 103.66 | 146.64 | 0.570 |
| Tree_d12 | 32.30 | 104.74 | 157.28 | 0.505 |
| Tree_d6 | 62.34 | 120.43 | 160.30 | 0.486 |
| Tree_d3 | 82.79 | 158.70 | 193.99 | 0.247 |
| Dummy | 119.57 | 194.52 | 254.61 | -0.296 |
**최종 Test (Train+Validation 재학습 후 1회)**: MAE 66.17 · RMSE 95.88 · R² 0.80
**해석과 한계**
- MAE 기준으로 RF_B를 선택했지만 RMSE·R²는 Linear가 더 좋았습니다. 선택은 "우선 지표"에 따른 결과입니다.
- RF_B 평균 잔차 -50.6 → 실제보다 높게 예측하는 경향
- 새벽(1–5시)과 출퇴근 피크(8시·17시)에서 오차가 큼
- 랜덤 포레스트는 학습 범위를 넘는 연도별 수요 증가를 외삽하지 못함
## Day 2. 온라인 쇼핑 구매 여부 예측 (분류 · PyTorch MLP)
- **데이터**: UCI Online Shoppers Purchasing Intention (12,330행 × 18열, 구매 15.47% 불균형)
- **분할**: Train 8,630 / Validation 1,850 / Test 1,850 (stratify, seed 42)
- **전처리**: 범주형 6개 OneHot(`handle_unknown="ignore"`), 수치형 10개 StandardScaler, `Weekend` 0/1
- **모델**: MLP 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1), Adam lr 0.001, batch 256
- **주 지표**: 구매 클래스 F1
| 모델 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| DummyClassifier | 0.845 | 0.000 | 0.000 | 0.000 |
| LogisticRegression | 0.882 | 0.734 | 0.367 | 0.490 |
| MLP-A (BCE) | 0.898 | 0.717 | 0.566 | 0.633 |
| **MLP-B (weighted BCE, pos_weight 5.46)** | 0.875 | 0.575 | **0.734** | **0.645** |
| MLP-B 임계값 | Precision | Recall | F1 | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.459 | 0.839 | 0.593 | 283 | 46 |
| **0.5** | 0.575 | 0.734 | **0.645** | 155 | 76 |
| 0.7 | 0.639 | 0.612 | 0.625 | 99 | 111 |
**최종 Test (Train+Validation, 17 epoch 재학습 후 1회)**: Accuracy 0.859 · Precision 0.527 · Recall 0.843 · F1 0.649
AI Mini Projects 가이드 · 4 / 70
혼동행렬 TN 1,348 / FP 216 / FN 45 / TP 241
**해석과 한계**
- Dummy도 Accuracy 0.845 → 불균형 데이터에서 Accuracy만으로는 평가 불가
- 가중 손실로 Recall을 높인 대신 FP가 늘어 Precision이 낮음
- `PageValues`가 실제 예측 시점에 사용 가능한 정보인지 추가 검토 필요
## 실행 방법
```bash
git clone https://github.com/Gracecodeit/ai-mini-projects.git
cd ai-mini-projects
uv sync
```
VS Code에서 노트북을 열고 커널로 `.venv`를 선택한 뒤 **Restart → Run All**을 실행합니다.
## 문서
| 문서 | 내용 |
|---|---|
| [01 데이터 분석](mini-project-01/docs/01_regression_data_analysis.md) | Day 1 EDA, 분할, 누수 확인 |
| [01 모델 분석](mini-project-01/docs/01_regression_model_analysis.md) | Day 1 모델 비교, 오류 분석 |
| [02 데이터 분석](mini-project-01/docs/02_classification_data_analysis.md) | Day 2 EDA, 계층화 분할, 전처리 |
| [02 모델 분석](mini-project-01/docs/02_classification_model_analysis.md) | Day 2 MLP 실험, 임계값 분석 |
| [ADR](mini-project-01/docs/adr.md) | 핵심 의사결정 ①–④ |
## Tech Stack
Python 3.11 · uv · pandas · NumPy · scikit-learn · PyTorch · matplotlib · joblib · Jupyter
=======
# uv 가상환경 설정

모든 명령어는 기존 `ai-mini-projects` 프로젝트의 최상위 폴더에서 실행합니다.

## 1. pyproject.toml 정리

기존 `uv init --package`로 생성된 `src/` 폴더를 삭제합니다. 단, 직접 작성한 코드가 없는 경우에만 삭제합니다.

`pyproject.toml`을 다음 내용으로 변경합니다.

```toml
[project]
name = "ai-mini-projects"
version = "0.1.0"
description = "멀티캠퍼스 AI 엔지니어 트랙 미니 프로젝트 모음"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []
```

기존 `[build-system]` 등 패키지 배포용 설정은 제거합니다. `README.md` 파일이 존재하는지도 확인합니다.

## 2. Python 버전 확인

기존에 생성된 `.python-version` 파일을 열어 다음과 같이 설정되어 있는지 확인합니다.

```text
3.11
```

이미 `3.11`이라면 별도의 설정 명령어는 필요하지 않습니다. 다른 버전이라면 `3.11`로 수정합니다.

## 3. 가상환경 생성 및 패키지 설치

```bash
uv add pandas numpy matplotlib scikit-learn ipykernel joblib
```

명령어를 실행하면 다음 작업이 자동으로 진행됩니다.

- `.venv/` — 가상환경 생성 및 패키지 설치
- `pyproject.toml` — 패키지 의존성 기록
- `uv.lock` — 패키지 버전 잠금

## 4. VS Code Jupyter 커널 설정

Jupyter Notebook을 열고 오른쪽 위 **Select Kernel → Python Environments**에서 `ai-mini-projects/.venv`에 해당하는 Python 3.11 환경을 선택합니다.

커널 이름은 환경에 따라 `ai-mini-projects`, `.venv` 등으로 표시될 수 있습니다.

## 5. Git 관리

`.gitignore`에 다음 항목이 포함되어 있는지 확인합니다.

```gitignore
.venv/
.ipynb_checkpoints/
```

**GitHub에 업로드할 파일**

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `.gitignore`
- `README.md`

`.venv/` 폴더는 GitHub에 업로드하지 않습니다.

환경 설정을 마쳤다면 해당 파일들을 Commit하고 Push합니다.

---

# AI Mini Projects

AI 엔지니어 과정에서 진행하는 미니프로젝트의
구현 과정과 실험 결과, 의사결정을 기록하는 저장소입니다.

## Mini Project 01

- 목표: AI 모델을 실험하고 결과를 분석하여
  적절한 방법을 선택하는 능력 기르기

  ### 프로젝트 자료

- [실험 코드](mini-project-01/code/)
- [실험 결과](mini-project-01/results/)
- [회귀 데이터 분석](mini-project-01/docs/01_regression_data_analysis.md)
- [회귀 모델 실험 및 오류 분석](mini-project-01/docs/01_regression_model_analysis.md)
- [분류 데이터 분석](mini-project-01/docs/02_classification_data_analysis.md)
- [분류 모델 실험 및 오류 분석](mini-project-01/docs/02_classification_model_analysis.md)
- [핵심 의사결정(ADR)](mini-project-01/docs/adr.md)

## Mini Project 02

- 목표: 사전학습된 모델의 성능을 직접 평가하고 실험 결과를 바탕으로 적절한 설정 또는 검색 방식을 선택하는 능력 기르기
- 진행 방식: CV와 NLP 중 하나를 선택하여 진행

### CV — YOLO 객체 탐지 모델의 성능 평가 및 최적 설정 선택

- [실험 코드](mini-project-02/code/CV_YOLO26_MiniProject.ipynb)
- [데이터 분석](mini-project-02/docs/01_cv_data_analysis.md)
- [모델 실험 및 오류 분석](mini-project-02/docs/01_cv_model_analysis.md)

### NLP — BERT 기반 한국어 의미 검색 시스템 개발 및 성능 비교

- [실험 코드](mini-project-02/code/NLP_BERT_MiniProject_Student.ipynb)
- [데이터 분석](mini-project-02/docs/02_nlp_data_analysis.md)
- [모델 실험 및 오류 분석](mini-project-02/docs/02_nlp_model_analysis.md)

### 공통 의사결정 기록

- [핵심 의사결정(ADR)](mini-project-02/docs/ADR.md)

### 문서 관리 원칙

- 실험별 상세 분석은 번호가 붙은 analysis 문서에 기록합니다.
- 주요 기술적 선택과 근거는 프로젝트별 ADR 문서에 요약합니다.
- 프로젝트 결과가 완성되면 이 README에 핵심 성과를 정리합니다.
>>>>>>> upstream/main
