# ADR — 핵심 의사결정 기록

**프로젝트:** 모듈 1 ML/DL 기초 미니프로젝트 — 회귀·분류

실험 결과를 반복해서 나열하기보다, 주요 기술적 선택과 선택 근거 및 한계를 기록한다. 1일차 회귀와 2일차 분류 프로젝트의 의사결정을 순서대로 정리한다.

---

## 1일차 — 회귀 (공유자전거 대여 수요 예측)

### 의사결정 ① 입력 특성 선택 (CP-02)

- **검토한 선택지:** 전체 컬럼 사용 또는 예측 시점에 활용할 수 있는 특성만 선별하는 방법을 비교했다. 특히 `casual`, `registered`의 정답 누수 여부와 `temp`, `atemp`의 중복 정보 가능성을 검토했다.
- **최종 선택:** `casual`, `registered`, `cnt`, `instant`, `dteday`를 제외하고 13개 특성을 사용했다. 근무일 출퇴근 시간대를 나타내는 `is_commute_hour`를 새로 생성했다.
- **선택 근거:** `casual + registered = cnt`가 모든 행에서 성립하므로 두 컬럼은 정답 누수에 해당하며, 예측 시점에도 알 수 없는 정보이므로 제외했다. 시간대별 대여 패턴을 반영하기 위해 근무일과 출퇴근 시간대를 결합한 특성을 추가했다.
- **감수한 한계 또는 추후 확인할 점:** `temp`와 `atemp`의 상관계수가 0.992로 높지만 두 특성을 모두 유지했다. 추후 각각 제거한 모델을 비교해 중복 정보가 예측 성능에 미치는 영향을 확인할 필요가 있다.

**분석 근거:** `01_regression_data_analysis.md`

### 의사결정 ② 최종 회귀 모델 선택 (CP-07)

- **비교한 모델·설정:** DummyRegressor, LinearRegression, Decision Tree(max_depth 3·6·12), Random Forest A(트리 50개·깊이 10), Random Forest B(트리 100개·깊이 15)를 비교했다.
- **최종 선택한 모델·설정:** `RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)`인 RF_B를 선택했다.
- **선택 근거:** RF_B의 Validation MAE가 96.63으로 비교 모델 중 가장 낮아 평균 절대 오차 기준에서 가장 적합하다고 판단했다. 다만 LinearRegression은 RMSE, R² 및 Train-Validation MAE 차이에서 더 나은 결과를 보여 모든 평가 지표에서 RF_B가 우수한 것은 아니었다.
- **선택 모델의 장점:** 비선형 관계와 특성 간 상호작용을 학습할 수 있으며, 비교 모델 중 Validation MAE가 가장 낮았다.
- **남아 있는 한계 또는 개선 방향:** Validation에서 일부 시간대의 예측 오차가 크게 나타났고 실제 대여량보다 높게 예측하는 경향도 확인했다. 시간대별 오류 원인을 추가 분석하고, 시간 순서를 보존하는 교차검증 및 하이퍼파라미터 탐색으로 개선할 필요가 있다.

**분석 근거:** `01_regression_model_analysis.md`

---

## 2일차 — 분류 (온라인 쇼핑 구매 여부 예측 · PyTorch MLP)

### 의사결정 ③ 평가 지표 선택 (CP-04)

- **검토한 선택지:** Accuracy, 구매 클래스 Precision, Recall, F1을 검토했다.
- **최종 선택한 주 지표와 보조 지표:** 구매 클래스(`Revenue=1`)의 F1을 주 평가 지표로 선택하고, Precision, Recall, Accuracy를 보조 지표로 사용했다.
- **선택 근거:** 구매 세션이 약 15%인 불균형 데이터에서 DummyClassifier는 모든 세션을 미구매로 예측해 Accuracy 0.845를 기록했지만 구매 클래스 F1은 0이었다. LogisticRegression은 Validation F1 0.490, Precision 0.734, Recall 0.367을 기록해 구매 고객 탐지 성능을 추가로 개선할 필요가 있었다.
- **감수한 한계 또는 추후 확인할 점:** F1은 Precision과 Recall의 균형을 평가하지만 FP와 FN의 실제 비용 차이를 직접 반영하지는 못한다. 실제 서비스에 적용할 경우 구매 고객을 놓치는 비용과 잘못된 구매 예측으로 발생하는 비용을 함께 고려해야 한다.

**분석 근거:** `02_classification_data_analysis.md`, `02_classification_model_analysis.md`

### 의사결정 ④ PyTorch MLP 설정 및 분류 임계값 선택 (CP-07)

- **검토한 선택지:** 기본 `BCEWithLogitsLoss`를 적용한 MLP-A와 양성 클래스 가중치(`pos_weight`)를 적용한 MLP-B를 비교했다. 각 모델의 분류 임계값 0.3, 0.5, 0.7에 따른 Precision, Recall, F1, FP, FN도 함께 확인했다.
- **최종 선택한 MLP 설정·임계값·재학습 에포크:** MLP-B(weighted BCE), 임계값 0.5를 선택했다. Train과 Validation을 합친 데이터로 17 epoch 재학습했다.
- **선택 근거:** MLP-B는 Validation F1 0.645로 MLP-A의 0.633보다 높았다. 임계값 0.5에서 Precision 0.575, Recall 0.734를 기록해 구매 고객 탐지율과 전체 F1의 균형이 가장 적절하다고 판단했다.
- **LogisticRegression과 비교했을 때 선택한 MLP의 특징:** 선택한 MLP-B의 Validation F1은 0.645로 LogisticRegression의 0.490보다 높았다. 이번 실험에서는 클래스 가중치를 적용한 신경망이 기준 선형 모델보다 구매 클래스 탐지 성능을 개선했다.
- **감수한 한계 또는 추후 개선 방향:** MLP-B는 Recall이 높아진 대신 FP가 증가해 Precision이 낮아졌다. 실제 적용 시에는 오탐 비용을 고려해 임계값을 조정하고, 다양한 임계값과 추가 검증 데이터에서 성능을 확인할 필요가 있다.

**분석 근거:** `02_classification_model_analysis.md`

---

**기록 원칙:** 최종 모델과 설정은 Validation 결과를 기준으로 결정하고, Test 데이터는 선택이 완료된 이후 최종 성능 평가에만 사용한다.
