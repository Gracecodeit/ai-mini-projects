# AI Mini Projects

멀티캠퍼스 AI 엔지니어 트랙 미니프로젝트 모음입니다.

- **모듈 1** 「[기본] AI 모델 실험과 검증」: 회귀·분류 모델 비교와 검증
- **모듈 2** 「Mini Project 02」: 사전학습 모델 성능 평가와 설정·검색 방식 선택 (NLP)

모델을 만드는 것보다 **여러 모델을 비교하고, 검증하고, 근거를 남겨 선택하는 과정**에 집중했습니다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)

## 한눈에 보기

| 프로젝트 | 주제 | 유형 | 최종 선택 | Test 결과 |
|---|---|---|---|---|
| Mini Project 01 · Day 1 | 공유자전거 시간당 대여량 예측 | 회귀 | RandomForestRegressor (100그루, 깊이 15) | MAE 66.17 / RMSE 95.88 / R² 0.80 |
| Mini Project 01 · Day 2 | 온라인 쇼핑 구매 여부 예측 | 이진 분류 | PyTorch MLP-B (weighted BCE, 임계값 0.5) | Recall 0.843 / F1 0.649 |
| Mini Project 02 · NLP | BERT 기반 한국어 의미 검색 | 문서 검색 | TF-IDF (문자 2–3-gram) | Hit@1 0.800 / Recall@3 0.800 / MRR@5 0.812 |

### 실험 원칙

- **데이터 누수 차단**: 정답을 그대로 만드는 컬럼(`casual + registered = cnt`)은 입력에서 제외
- **올바른 분할**: Day 1은 날짜순 분할(미래 정보 차단), Day 2는 `stratify` 계층화 분할(구매 비율 유지)
- **Train에만 fit**: 스케일러·인코더 기준은 Train에서만 학습하고 Validation/Test에는 transform만 적용
- **Baseline부터 비교**: Dummy → Linear/Logistic → Tree/RF/MLP 순으로 성능 향상 여부 확인
- **Test는 한 번만**: 모델·임계값·검색 방식은 Validation으로 결정하고 Test는 최종 확인에만 사용
- **의사결정 기록**: 지표·모델·임계값 선택 근거를 ADR에 정리 ([Mini Project 01 ADR](mini-project-01/docs/adr.md), [Mini Project 02 ADR](mini-project-02/docs/ADR.md))

## 폴더 구조

```
ai-mini-projects/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── mini-project-01/
│   ├── code/
│   │   ├── 01_regression_bike_sharing.ipynb
│   │   └── 02_classification_online_shopping.ipynb
│   ├── data/
│   │   ├── hour.csv
│   │   └── online_shoppers_intention.csv
│   ├── results/
│   │   ├── 01_model_comparison.csv
│   │   ├── 01_final_model.joblib
│   │   ├── 02_model_comparison.csv
│   │   ├── 02_threshold_comparison.csv
│   │   ├── 02_model_config.json
│   │   └── 02_final_classifier.joblib
│   └── docs/
│       ├── 01_regression_data_analysis.md
│       ├── 01_regression_model_analysis.md
│       ├── 02_classification_data_analysis.md
│       ├── 02_classification_model_analysis.md
│       └── adr.md
└── mini-project-02/
    ├── code/
    │   ├── CV_YOLO26_MiniProject.ipynb
    │   └── NLP_BERT_MiniProject_Student.ipynb
    └── docs/
        ├── 01_cv_data_analysis.md
        ├── 01_cv_model_analysis.md
        ├── 02_nlp_data_analysis.md
        ├── 02_nlp_model_analysis.md
        └── ADR.md
```

---

# Mini Project 01

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

혼동행렬 TN 1,348 / FP 216 / FN 45 / TP 241

**해석과 한계**

- Dummy도 Accuracy 0.845 → 불균형 데이터에서 Accuracy만으로는 평가 불가
- 가중 손실로 Recall을 높인 대신 FP가 늘어 Precision이 낮음
- `PageValues`가 실제 예측 시점에 사용 가능한 정보인지 추가 검토 필요

---

# Mini Project 02

- 목표: 사전학습된 모델의 성능을 직접 평가하고 실험 결과를 바탕으로 적절한 설정 또는 검색 방식을 선택하는 능력 기르기
- 진행 방식: CV와 NLP 중 하나를 선택하여 진행

## CV — YOLO 객체 탐지 모델의 성능 평가 및 최적 설정 선택

- [실험 코드](mini-project-02/code/CV_YOLO26_MiniProject.ipynb)
- [데이터 분석](mini-project-02/docs/01_cv_data_analysis.md)
- [모델 실험 및 오류 분석](mini-project-02/docs/01_cv_model_analysis.md)

## NLP — BERT 기반 한국어 의미 검색 시스템 개발 및 성능 비교

KLUE-MRC의 질문–문단 데이터를 **검색 문제**로 재구성해 TF-IDF, `klue/bert-base`, 한국어 Sentence-BERT 세 가지 검색 방식을 직접 구현하고 비교했습니다. GPT로 답변을 생성하거나 BERT를 재학습하지 않으며, 문서 표현 → 유사도 계산 → Top-k 검색 → 정답 평가 → 오류 분석 → 최종 검색 방식 선정을 다룹니다.

- [실험 코드](mini-project-02/code/NLP_BERT_MiniProject_Student.ipynb)
- [데이터 분석](mini-project-02/docs/02_nlp_data_analysis.md)
- [모델 실험 및 오류 분석](mini-project-02/docs/02_nlp_model_analysis.md)

### 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 입력 / 출력 | 한국어 질문 1개 / 유사도 순 문서 Top-k (`rank`, `doc_id`, `score`, `text`) |
| 데이터 | KLUE-MRC(`klue/klue`, `mrc`, validation split)에서 재구성한 교육용 검색 세트 |
| 구성 | 문단 200개 / Validation 질문 60개 / 보류 Test 질문 20개 (질문마다 정답 문서 1개) |
| 문서 길이 | 498–500자 (평균 499.7), 질문 길이 13–54자 (평균 29.2) |
| 평가 지표 | Hit@1, Recall@3, MRR@5 (Top-5 검색 결과 기준) |
| 재현성 | 시드 42, 임베딩 캐시를 지우고 처음부터 다시 실행해도 동일한 수치 확인 |

> 수업용 단일 정답 문단 검색 세트이며 KLUE-MRC의 공식 검색 벤치마크가 아닙니다. 공식 KLUE-MRC 점수와 비교할 수 없습니다.

### 검색 방식

| 방식 | 문서 표현 | 유사도 |
|---|---|---|
| TF-IDF | 문자 2–3-gram, `max_features=30000` | 코사인 유사도 |
| BERT | `klue/bert-base` 토큰 벡터 + Attention Mask Mean Pooling (최대 256토큰, 768차원) | 직접 구현한 코사인 유사도 |
| Sentence-BERT | `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (최대 256토큰, 768차원) | 코사인 유사도 |

### Validation 결과 (질문 60개)

| 검색 방식 | Hit@1 | Recall@3 | MRR@5 |
|---|---:|---:|---:|
| **TF-IDF** | **0.883** | **0.967** | **0.917** |
| BERT | 0.667 | 0.833 | 0.759 |
| Sentence-BERT | 0.767 | 0.883 | 0.821 |

- **활용 상황**: 관련 문서 후보 3개를 보여 주는 한국어 문서 검색기
- **우선 지표**: Recall@3 (후보 3개 안에 정답이 들어 있는지가 핵심)
- **최종 선택**: `tfidf`. 우선 지표를 포함한 세 지표 모두 가장 높고, 모델 로딩과 임베딩 계산이 필요 없어 실행 비용도 가장 낮을 것으로 보입니다(처리 시간은 직접 측정하지 않았습니다).

### 보류 Test 결과 (질문 20개, 선택 방식 고정 후 1회 평가)

| 데이터 | 질문 수 | Hit@1 | Recall@3 | MRR@5 |
|---|---:|---:|---:|---:|
| Validation | 60 | 0.883 | 0.967 | 0.917 |
| Test | 20 | 0.800 | 0.800 | 0.812 |

**해석과 한계**

- Test가 20개뿐이라 질문 1개가 Hit@1·Recall@3을 0.05씩 바꿉니다. 차이를 실제 성능 저하로 단정하기 어렵습니다. Hit@1과 Recall@3이 같으므로 1위를 놓친 4개 질문은 모두 상위 3개 밖이었습니다.
- 예상(겹침이 낮은 질문에서는 BERT 계열이 유리)과 달리 전체 성능은 TF-IDF가 가장 높았습니다. 질문과 정답 문단이 표현을 일부 공유하고 Corpus가 작았기 때문일 수 있으나 검증하지는 않았습니다.
- 표현 차이가 큰 질문(예: QV047)은 TF-IDF가 Top-5에서 찾지 못했고 Sentence-BERT만 4위로 찾았습니다. 의미 기반 방식과의 결합은 추후 검토 대상입니다.
- 같은 Corpus에서 질문만 바꿔 평가했고 질문마다 정답 문서를 1개로 지정했으므로, 이 결과로 실제 업무 검색 품질을 보장할 수 없습니다.

```python
for r in search_docs('질문을 입력하세요', k=3):
    print(r['rank'], r['doc_id'], r['score'], r['text'][:60])
```

동작 확인 예시: "오라클과 구글의 저작권 소송에서 미국 항소법원은 어떤 판단을 내렸나요?"를 입력하면 1위로 `D048`(0.4736)이 검색됩니다. 이 질문은 해당 문서를 보고 만든 것이라 성능 지표가 아니라 동작 확인용입니다.

### 공통 의사결정 기록

- [핵심 의사결정(ADR)](mini-project-02/docs/ADR.md)

---

## 실행 방법

### Mini Project 01

```bash
git clone https://github.com/Gracecodeit/ai-mini-projects.git
cd ai-mini-projects
uv sync
```

VS Code에서 노트북을 열고 커널로 `.venv`를 선택한 뒤 **Restart → Run All**을 실행합니다.

### Mini Project 02 (NLP)

1. `mini-project-02/code/NLP_BERT_MiniProject_Student.ipynb`를 열고 **00번 환경 설치 셀부터 순서대로** 실행합니다. 패키지, 데이터, 모델 가중치는 없으면 자동으로 준비되고 저장된 파일은 재사용합니다. NLP 노트북은 `uv sync` 대신 노트북 안의 pip 설치 셀로 패키지를 준비합니다.
2. Day 1(CP-01–CP-06)에서 데이터 분석과 TF-IDF·BERT 검색기를 구현하고, Day 2(CP-07–CP-10)에서 평가·비교·오류 분석을 거쳐 `search_docs()`를 완성합니다.
3. CP-01–CP-09는 Validation 질문만 사용하며, **Test 정답은 CP-10에서 처음 사용**합니다.
4. 실행 환경은 Python 3.12.3, PyTorch 2.8.0+cu128, NVIDIA GeForce RTX 4090(GPU)이었으며, GPU가 없으면 CPU로 실행되어 BERT 임베딩 계산이 더 오래 걸립니다. 주요 패키지는 `transformers>=4.46,<5`, `sentence-transformers>=3.0,<6`, `scikit-learn>=1.3`입니다.
5. RunPod처럼 `HF_HUB_ENABLE_HF_TRANSFER=1`이 켜진 환경에서는 `hf_transfer` 패키지가 필요합니다. 노트북의 데이터 준비 앞 셀이 이를 설치합니다.
6. 원본 데이터와 대형 모델 가중치는 GitHub에 올리지 않습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [01 데이터 분석](mini-project-01/docs/01_regression_data_analysis.md) | Day 1 EDA, 분할, 누수 확인 |
| [01 모델 분석](mini-project-01/docs/01_regression_model_analysis.md) | Day 1 모델 비교, 오류 분석 |
| [02 데이터 분석](mini-project-01/docs/02_classification_data_analysis.md) | Day 2 EDA, 계층화 분할, 전처리 |
| [02 모델 분석](mini-project-01/docs/02_classification_model_analysis.md) | Day 2 MLP 실험, 임계값 분석 |
| [Mini Project 01 ADR](mini-project-01/docs/adr.md) | 핵심 의사결정 ①–④ |
| [NLP 데이터 분석](mini-project-02/docs/02_nlp_data_analysis.md) | 검색 데이터셋 구조, 길이·겹침 분석, 초기 가설 |
| [NLP 모델 분석](mini-project-02/docs/02_nlp_model_analysis.md) | 세 검색기 비교, 오류 분석, Test 평가 |
| [Mini Project 02 ADR](mini-project-02/docs/ADR.md) | 활용 목적·우선 지표·최종 검색 방식 의사결정 |

## Tech Stack

Python 3.11 · uv · pandas · NumPy · scikit-learn · PyTorch · Hugging Face Transformers · sentence-transformers · matplotlib · joblib · Jupyter