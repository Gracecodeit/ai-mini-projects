# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** 구매 클래스 F1
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교 데이터:** Train 8,630행 / Validation 1,850행
- **DummyClassifier — Validation Accuracy / F1:** **0.845 / 0.000**
- **LogisticRegression — Train F1 / Validation F1:** **0.505 / 0.490**
- **LogisticRegression — Validation Precision / Recall:** **0.734 / 0.367**
- **LogisticRegression Validation 혼동행렬:** TN **1,526** / FP **38** / FN **181** / TP **105**

**분석:**  
구매하지 않은 고객이 전체의 다수를 차지하기 때문에 DummyClassifier는 모든 데이터를 미구매로 예측해도 Accuracy 0.845를 기록했다. 그러나 구매 고객을 한 명도 찾지 못해 구매 클래스 F1은 0이었다. 따라서 불균형 데이터에서는 Accuracy만으로 모델 성능을 판단하기 어렵고, 구매 고객 탐지 성능을 함께 평가해야 한다.

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1 최고 에포크:** 25 (최대 학습 에포크와 같아, 더 학습하면 개선될 여지가 있음)
- **MLP-A Train F1 / Validation F1:** **0.709 / 0.633**
- **MLP-A Validation Precision / Recall:** **0.717 / 0.566**

**분석:**  
MLP-A는 LogisticRegression보다 Validation F1이 0.490에서 0.633으로 향상되었다. 다만 Train F1(0.709)이 Validation F1(0.633)보다 높아 학습 데이터와 검증 데이터 사이에 성능 차이가 나타났다. 마지막 5개 에포크(21~25)에서 Train loss는 0.217에서 0.212로, Validation F1은 0.618에서 0.633으로 완만하게 변했고, 최고 Validation F1 에포크가 학습 상한인 25였다.

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터 `pos_weight`:** **5.46**
- **MLP-B 최적 에포크:** **17**
- **MLP-B Train F1 / Validation F1:** **0.692 / 0.645**
- **MLP-B Validation Precision / Recall:** **0.575 / 0.734**
- **MLP-A와 MLP-B의 Validation F1 차이:** **MLP-B가 0.012 높음**
- **모델 비교 결과:** MLP-B가 Validation F1 0.645로 가장 높았으며, LogisticRegression(0.490)과 MLP-A(0.633)를 앞섰다.
- **Train·Validation F1 차이:** MLP-A **0.077**, MLP-B **0.047**

**분석:**  
양성 클래스 가중치를 적용한 MLP-B는 MLP-A보다 Precision이 0.717에서 0.575로 낮아졌지만, Recall은 0.566에서 0.734로 높아졌다. 구매 고객을 더 많이 탐지하는 대신 오탐이 증가한 것이다. Validation F1도 0.633에서 0.645로 소폭 향상되었으며, Train과 Validation의 F1 차이도 줄어들었다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교 모델:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A FP / FN:** **64 / 124**
- **기본 임계값 0.5에서 MLP-B FP / FN:** **155 / 76**

| 모델 / 임계값 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
| MLP-A / 0.3 | 0.608 | 0.678 | 0.641 | 125 | 92 |
| MLP-A / 0.5 | 0.717 | 0.566 | 0.633 | 64 | 124 |
| MLP-A / 0.7 | 0.835 | 0.353 | 0.496 | 20 | 185 |
| MLP-B / 0.3 | 0.459 | 0.839 | 0.593 | 283 | 46 |
| MLP-B / 0.5 | 0.575 | 0.734 | **0.645** | 155 | 76 |
| MLP-B / 0.7 | 0.639 | 0.612 | 0.625 | 99 | 111 |

- **임계값 0.3:** 구매 판정 기준이 낮아져 Recall이 높아지고 FN은 감소하지만, FP가 증가한다.
- **임계값 0.7:** 구매 판정 기준이 높아져 Precision이 높아지고 FP는 감소하지만, FN이 증가한다.
- **Validation F1 최고 조합:** **MLP-B / 임계값 0.5 (F1 0.645)**
- **선택 모델의 최적 에포크:** **17**
- **추가 관찰:** MLP-A는 임계값 0.3에서 F1 0.641을 기록해 MLP-B와 성능 차이가 크지 않았다.

**분석:**  
임계값을 낮추면 구매 고객을 더 많이 찾을 수 있지만 실제 미구매 고객을 구매자로 판단하는 FP가 증가한다. 반대로 임계값을 높이면 FP는 줄지만 구매 고객을 놓치는 FN이 증가한다. 이번 실험에서는 구매 고객 탐지와 전체적인 F1의 균형을 고려해 MLP-B와 임계값 0.5를 선택했다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 평가 전 확정한 모델 / 임계값:** **MLP-B (weighted BCE) / 0.5**
- **최종 재학습 데이터:** **10,480행 (Train + Validation)**
- **최종 재학습 에포크 / 가중 손실:** **17 / 적용**
- **Test Accuracy / Precision / Recall / F1:** **0.859 / 0.527 / 0.843 / 0.649**
- **Test 혼동행렬:** TN **1,348** / FP **216** / FN **45** / TP **241**
- **모델 번들·설정 JSON 저장 및 재로딩 확인:** 저장 전후 Test 예측 일치 `True`
- **`predict_purchase()` 실행 결과:** Test 첫 번째 세션의 구매 확률 0.0058, 예측 0 (임계값 0.5), 실제 `Revenue` 0

**해석:**  
최종 Test F1은 0.649로 Validation F1 0.645와 유사하게 나타났다. Recall은 Validation 0.734에서 Test 0.843으로 높아졌고, Precision은 0.575에서 0.527로 낮아졌다. Test에서도 구매 고객을 비교적 잘 탐지했지만 FP가 216건 발생해 구매 예측의 정확성을 개선할 여지가 남아 있다. Test 데이터는 최종 모델 선택 이후 평가에만 사용했다.

## 6. 실험 요약

- **가장 중요한 실험 결과:** 양성 클래스 가중치를 적용한 MLP-B가 Validation F1 0.645, Test F1 0.649를 기록했다. 클래스 가중치와 임계값 조정을 통해 구매 고객 탐지 성능을 개선할 수 있었다.
- **남아 있는 모델의 한계:** 구매 고객 탐지율은 높지만 FP가 여전히 발생해 실제 구매 가능성이 낮은 고객을 구매자로 예측하는 문제가 남아 있다. 또한 Train과 Validation의 성능 차이 및 입력 변수의 실제 예측 시점 활용 가능성을 추가로 검토할 필요가 있다.

> **최종 MLP 설정·임계값의 선택 근거**는 `adr.md`의 의사결정 ④에 별도로 기록한다.