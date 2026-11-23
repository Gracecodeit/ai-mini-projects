# AI Mini Projects

멀티캠퍼스 AI 엔지니어 과정에서 진행하는 미니프로젝트의 코드, 데이터 분석, 모델 실험 결과 및 기술적 의사결정을 관리하는 저장소입니다.

## 프로젝트 구성

| 프로젝트 | 학습 주제 | 진행 방식 |
| --- | --- | --- |
| [Mini Project 01](mini-project-01/) | ML/DL 기초 — 회귀 및 분류 | 회귀·분류 모두 진행 |
| [Mini Project 02](mini-project-02/) | CV/NLP 모델링 — 객체 탐지 및 의미 기반 검색 | CV·NLP 중 하나 선택 |

## Mini Project 01 — ML/DL 기초

데이터를 분석하고 여러 모델과 설정을 비교하여, 실험 결과를 바탕으로 적절한 모델을 선택하는 과정을 학습합니다.

### 1. 공유자전거 대여 수요 예측 (회귀)

- 데이터: UCI Bike Sharing Dataset (`hour.csv`)
- 주요 내용: 데이터 탐색, 시간 순서 분할, 특성공학, 회귀 모델 비교, 예측 오류 분석
- 평가 지표: MAE, RMSE, R²

| 프로젝트 자료 | 링크 |
| --- | --- |
| 실습 Notebook | [01_regression_bike_demand.ipynb](mini-project-01/code/01_regression_bike_demand.ipynb) |
| 데이터 분석 | [01_regression_data_analysis.md](mini-project-01/docs/01_regression_data_analysis.md) |
| 모델 실험 및 오류 분석 | [01_regression_model_analysis.md](mini-project-01/docs/01_regression_model_analysis.md) |

### 2. 온라인 쇼핑 구매 여부 예측 (분류)

- 데이터: UCI Online Shoppers Purchasing Intention Dataset
- 주요 내용: 클래스 불균형 분석, 데이터 전처리, PyTorch MLP 구현, 가중 손실 및 분류 임계값 비교
- 평가 지표: Accuracy, Precision, Recall, F1

| 프로젝트 자료 | 링크 |
| --- | --- |
| 실습 Notebook | [02_classification_online_shopping.ipynb](mini-project-01/code/02_classification_online_shopping.ipynb) |
| 데이터 분석 | [02_classification_data_analysis.md](mini-project-01/docs/02_classification_data_analysis.md) |
| 모델 실험 및 오류 분석 | [02_classification_model_analysis.md](mini-project-01/docs/02_classification_model_analysis.md) |

**공통 기술적 의사결정 기록:** [adr.md](mini-project-01/docs/adr.md)

## Mini Project 02 — CV/NLP 모델링

사전학습된 모델을 활용하여 평가 알고리즘을 직접 구현하고, 실험 및 오류 분석 결과를 바탕으로 활용 목적에 적합한 방법을 선택합니다.

CV와 NLP 중 하나를 선택하여 이틀 동안 CP-01~CP-10을 진행합니다.

### 1. CV — YOLO 객체 탐지 모델의 성능 평가 및 최적 설정 선택

- 데이터: COCO 2017 Validation에서 선별한 이미지 160개 (Validation 120개, Test 40개)
- 모델: YOLO26 Nano (`yolo26n.pt`)
- 탐지 대상: Person, Bicycle, Car, Bus
- 주요 내용: 객체 크기 분석, IoU 및 1:1 매칭 구현, Confidence Threshold별 성능 비교, 탐지 오류 분석
- 평가 지표: Precision, Recall, F1 (평가용 IoU Threshold 0.5)
- 최종 결과물: 객체 탐지 평가기, 장면 분석기 및 최종 설정의 Test 평가 결과

| 프로젝트 자료 | 링크 |
| --- | --- |
| 실습 Notebook | [CV_YOLO26_MiniProject.ipynb](mini-project-02/code/CV_YOLO26_MiniProject.ipynb) |
| 데이터 분석 | [01_cv_data_analysis.md](mini-project-02/docs/01_cv_data_analysis.md) |
| 모델 실험 및 오류 분석 | [01_cv_model_analysis.md](mini-project-02/docs/01_cv_model_analysis.md) |

### 2. NLP — BERT 기반 한국어 의미 검색 시스템 개발 및 성능 비교

- 데이터: KLUE-MRC에서 재구성한 검색 데이터 (문단 200개, Validation 질문 60개, Test 질문 20개)
- 검색 방식: TF-IDF, BERT + Masked Mean Pooling, Sentence-BERT
- 주요 내용: 데이터 특성 분석, 문서 임베딩 생성, 코사인 유사도 및 Top-k 검색 구현, 검색 성능 비교 및 오류 분석
- 평가 지표: Hit@1, Recall@3, MRR@5
- 최종 결과물: 문서 검색 함수 `search_docs()` 및 최종 검색 방식의 Test 평가 결과

| 프로젝트 자료 | 링크 |
| --- | --- |
| 실습 Notebook | [NLP_BERT_MiniProject_Student.ipynb](mini-project-02/code/NLP_BERT_MiniProject_Student.ipynb) |
| 데이터 분석 | [02_nlp_data_analysis.md](mini-project-02/docs/02_nlp_data_analysis.md) |
| 모델 실험 및 오류 분석 | [02_nlp_model_analysis.md](mini-project-02/docs/02_nlp_model_analysis.md) |

**공통 기술적 의사결정 기록:** [ADR.md](mini-project-02/docs/ADR.md)

ADR에는 CV와 NLP의 작성 양식이 모두 포함되어 있습니다. 선택하지 않은 프로젝트의 양식을 삭제하고 선택한 프로젝트에 해당하는 내용만 작성합니다.

## 실행 환경 설정

### 1. uv 가상환경 설정 (로컬 PC)

모든 터미널 명령어는 `ai-mini-projects` 저장소의 최상위 폴더에서 실행합니다.

이미 `pyproject.toml`, `uv.lock`, `.python-version`이 구성되어 있다면 기존 설정을 변경하지 않고 다음 명령어로 가상환경과 패키지를 준비합니다.

```bash
uv sync
```

처음 환경을 구성하는 경우 아래 절차를 참고합니다.

#### 1-1. pyproject.toml 정리

기존 `uv init --package`로 생성한 `src/` 폴더는 직접 작성한 코드가 없는 경우에만 삭제합니다.

새로 프로젝트를 설정하는 경우 `pyproject.toml`의 기본 구성은 다음과 같습니다.

```toml
[project]
name = "ai-mini-projects"
version = "0.1.0"
description = "멀티캠퍼스 AI 엔지니어 트랙 미니 프로젝트 모음"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []
```

패키지 배포용으로 생성한 프로젝트라면 불필요한 `[build-system]` 등 배포 설정을 정리합니다. 이미 등록된 의존성은 지우지 않습니다.

#### 1-2. Python 버전 확인

`.python-version` 파일에 다음과 같이 설정되어 있는지 확인합니다.

```text
3.11
```

#### 1-3. 가상환경 생성 및 패키지 설치

```bash
uv add pandas numpy matplotlib scikit-learn ipykernel joblib
```

PyTorch 분류 실습에 필요한 PyTorch가 설치되지 않은 경우 추가합니다.

```bash
uv add torch
```

`uv add`는 가상환경에 패키지를 설치하고 `pyproject.toml`과 `uv.lock`을 갱신합니다.

#### 1-4. VS Code Jupyter 커널 설정

Notebook을 열고 오른쪽 위 **Select Kernel → Python Environments**에서 `ai-mini-projects/.venv`의 Python 3.11 환경을 선택합니다.

커널 이름은 환경에 따라 `ai-mini-projects`, `.venv` 등으로 표시될 수 있습니다.

### 2. Mini Project 02 실행 환경

CV/NLP 미니프로젝트는 GPU 환경 사용을 권장합니다. Notebook의 첫 환경 준비 셀부터 순서대로 실행하면 필요한 패키지, 데이터, 모델 가중치 및 작업 폴더를 준비할 수 있습니다.

- Day 1: CP-01~CP-06 완료 후 작업한 Notebook을 GitHub에 Push합니다.
- Day 2: 저장한 Notebook을 다시 받아 환경 준비부터 순서대로 실행하고 CP-07~CP-10을 완료합니다.
- GPU 세션 초기화로 작업 폴더가 사라진 경우 데이터와 모델을 다시 준비해야 합니다. 기존 파일이 남아 있다면 Notebook에서 지원하는 캐시를 재사용할 수 있습니다.

## 문서 작성 및 제출

각 프로젝트의 안내문과 Notebook을 기준으로 작성합니다.

| 문서 | 작성 내용 |
| --- | --- |
| Data Analysis | 데이터 특성, 분석 결과 및 실험 전 가설 |
| Model Analysis | 실제 성능 수치, 실험 결과, 오류 분석 및 최종 평가 |
| ADR (Architecture Decision Record) | 주요 기술적 선택, 선택 근거 및 트레이드오프 |

Mini Project 01은 회귀·분류의 분석 문서와 공통 `adr.md`를 작성합니다. Mini Project 02는 선택한 분야의 데이터 분석 문서, 모델 분석 문서, 공통 `ADR.md`를 작성합니다. 별도 프로젝트 명세서는 수업에서 안내한 양식을 사용합니다.

실험 및 모델 선택에는 Validation 데이터를 사용하고, Test 데이터는 최종 평가에만 사용합니다. 분석 문서에는 실제 Notebook의 실행 결과를 기록합니다.

## Git 관리

`.gitignore`에 다음 항목이 포함되어 있는지 확인합니다.

```gitignore
.venv/
.ipynb_checkpoints/
__pycache__/
```

- GitHub에 관리할 파일: 실습 Notebook, 분석 문서, ADR, `README.md`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore` 등 프로젝트에 필요한 소스와 설정 파일
- 업로드하지 않을 파일: 가상환경, 미니프로젝트 2의 다운로드한 원본 데이터·대용량 모델 가중치·임베딩 및 추론 캐시

필요한 실험 결과는 분석 문서에 정리하고, 제출에 필요한 파일을 Commit한 뒤 Push합니다.
