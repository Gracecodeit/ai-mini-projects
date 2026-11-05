# ADR — 주요 기술 의사결정 기록

## 의사결정 ① 입력 특성 선택

### 1. 결정 배경 (Context)

공유자전거의 시간당 총 대여량인 `cnt`를 예측하기 위해 어떤 컬럼을 입력으로 사용할지 정해야 했다. 모든 컬럼을 그대로 사용하면 예측 시점에 알 수 없는 정보가 포함되어 평가 결과가 실제보다 좋아질 수 있으므로, 각 컬럼을 실제 예측 상황에서 사용할 수 있는지 확인했다.

### 2. 검토한 대안 (Alternatives)

- 대안 A: 데이터에 있는 모든 컬럼을 입력 특성으로 사용한다.
- 대안 B: 정답과 직접 연결되어 있거나 예측 시점에 알 수 없는 컬럼을 제외하고, 사용할 수 있는 특성만 입력한다.

### 3. 최종 결정 (Decision)

대안 B를 선택했다. `casual`, `registered`, `cnt`, `instant`, `dteday`를 모델 입력에서 제외하고, `is_commute_hour`를 새로 만들어 총 13개 특성을 사용했다.

사용한 특성은 `season`, `yr`, `mnth`, `hr`, `weekday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `holiday`, `workingday`, `is_commute_hour`이다.

### 4. 결정 근거

`casual`과 `registered`의 합에서 `cnt`를 뺀 값의 최대 절댓값이 0이었다. 두 컬럼의 합이 정답과 정확히 같으므로 입력으로 사용하면 데이터 누수가 발생한다. 또한 두 값은 해당 시간의 대여가 끝나야 알 수 있어 예측 시점에 사용할 수 없으므로 제외했다.

`instant`는 행 번호이고 `dteday`는 날짜 문자열이므로 모델 입력에서는 제외했다. 날짜 정보는 날짜순 데이터 분할에 사용했다.

시간대별 그래프에서 근무일 출퇴근 시간대에 대여량이 증가하는 패턴이 보여, 근무일이면서 7\~9시 또는 17\~19시인 경우 1이 되는 `is_commute_hour`를 추가했다. `temp`와 `atemp`의 상관계수는 0.992로 매우 높았지만 이번 실험에서는 두 특성을 모두 유지했다.

### 5. 트레이드오프 및 영향 (Trade-off & Consequence)

- 얻는 이점: 예측 시점에 알 수 없는 누수 정보를 제거하고, 출퇴근 시간과 근무일의 조합 정보를 모델에 제공할 수 있다.
- 감수한 한계: `temp`와 `atemp`처럼 정보가 유사한 특성을 함께 사용했다. 둘 중 하나를 제거했을 때 성능이 어떻게 달라지는지는 별도로 검증하지 않았다.
- 후속 조치: 필요하다면 두 온도 특성을 각각 사용한 모델을 비교해 중복 정보가 예측 성능과 선형회귀 계수에 미치는 영향을 확인한다.

## 의사결정 ② 최종 모델 선택

### 1. 결정 배경 (Context)

DummyRegressor, LinearRegression, Decision Tree, Random Forest의 성능을 비교한 뒤 최종 모델을 선택해야 했다. 평균적인 예측 오차뿐 아니라 큰 오차, 설명력, Train과 Validation의 차이도 함께 확인했다.

### 2. 검토한 대안 (Alternatives)

- Dummy: Train 평균을 예측하는 기준 모델
- Linear: 입력 특성과 대여량의 선형 관계를 학습하는 모델
- Decision Tree: 깊이 3, 6, 12로 설정한 단일 트리
- Random Forest A: 트리 50개, 최대 깊이 10
- Random Forest B: 트리 100개, 최대 깊이 15

### 3. 최종 결정 (Decision)

`RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)`인 RF_B를 최종 모델로 선택했다. 선택 후 Train과 Validation 데이터를 합쳐 재학습하고, Test 데이터로 최종 성능을 한 번 평가했다.

### 4. 선택 근거

RF_B의 Validation MAE는 96.63으로 비교한 모델 중 가장 낮았다. Linear는 99.97로 약 3.34대 차이였고, Dummy는 194.52였다. 이번 프로젝트에서는 평균 절대 오차인 MAE를 주된 선택 기준으로 삼았기 때문에 RF_B를 선택했다.

다만 RF_B가 모든 지표에서 우수한 것은 아니다. Linear의 Validation RMSE는 124.36으로 RF_B의 144.08보다 낮았고, R²도 Linear가 0.691, RF_B가 0.585로 더 높았다. MAE Gap 역시 Linear가 44.63으로 RF_B의 73.72보다 작았다. 즉 Linear는 큰 오차와 Train-Validation 차이 측면에서 더 나은 모습을 보였고, RF_B는 평균 절대 오차에서 조금 더 나은 결과를 보였다.

RF_B의 평균 잔차(실제값-예측값)는 -50.6으로 Validation에서 실제보다 높게 예측하는 경향이 있었다. 시간대별 MAE도 새벽 3시 170.5, 오전 8시 123.5, 오후 17시 119.7로 해당 시간대에서 큰 오차가 확인되었다.

Train과 Validation의 MAE 차이가 크지만, Dummy의 MAE Gap도 74.95였다. 따라서 이 차이를 전부 RF_B의 과대적합으로만 해석할 수는 없다. Validation 기간의 수요 수준이 Train과 달랐을 가능성도 함께 고려했다.

### 5. 결과 및 트레이드오프

최종 RF_B를 Train과 Validation으로 다시 학습한 뒤 Test에서 다음 성능을 얻었다.

| 평가 지표 | Test 결과 |
|---|---:|
| MAE | 66.17 |
| RMSE | 95.88 |
| R² | 0.80 |

RF_B를 선택해 비교 모델 중 가장 낮은 Validation MAE를 얻었다. 반면 Linear보다 RMSE와 R²가 낮고 MAE Gap이 크다는 한계가 남았다. Test MAE가 Validation MAE보다 낮게 나온 이유는 학습 데이터가 늘어난 점이나 두 평가 기간의 수요 분포 차이 등 여러 가능성이 있지만, 각 요인의 영향을 따로 검증하지는 않았다.

### 6. 남은 과제

- 새벽 시간대의 큰 오차가 특정 날짜, 주말, 계절 또는 이상값과 관련 있는지 추가 분석한다.
- 큰 오차를 더 중요하게 보는 경우 RMSE를 기준으로 모델 선택 결과가 달라지는지 비교한다.
- 시간 순서를 보존하는 `TimeSeriesSplit`과 `GridSearchCV`를 사용해 모델 설정을 탐색한다.
- 연도에 따른 수요 증가처럼 학습 데이터의 범위를 넘어서는 추세를 어떻게 반영할지 검토한다.

## 관련 분석 문서

- [데이터 분석](01_regression_data_analysis.md)
- [실험 분석](01_regression_model_analysis.md)
