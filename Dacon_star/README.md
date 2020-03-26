# [과학] 월간 데이콘 2 천체 유형 분류 대회
link : <https://dacon.io/competitions/official/235573/overview/>
## 
#### Free stage
* Baseline 그대로 통과
## 
#### Regular Stage 1
code : <https://github.com/KaJaeHyeob/DeepLearning/blob/master/Dacon_star/submission/0384-0.3688.py>
* 아웃라이어 제거를 위해 IQR 탐색, LocalOutlierFactor(), IsolationForest() 사용
  - IsolationForest() 가 가장 효과가 좋았음.
  - 데이터만 봐선 어떤 방법이 가장 효과적인지 알기 힘듬. 모두 해보는 것이 좋음.
* 잡히지 않은 아웃라이어의 영향을 최소화하기 위해 정규화 과정에서 RobustScaler() 사용
## 
#### Regular Stage 2
code : <https://github.com/KaJaeHyeob/DeepLearning/blob/master/Dacon_star/submission/0234-0.3492.py>
* 정규화 과정으로 생겨난 음수값으로 인해 발생되는 Gradient Vanishing 방지를 위해 활성화 함수를 relu 에서 mish 로 대체
## 
#### Final Stage
code : <https://github.com/KaJaeHyeob/DeepLearning/blob/master/Dacon_star/submission/0241-0.3586.py>
* 아웃라이어 제거 대신 평균값 또는 상하한으로 대체
  - 결과가 좋지 못했음. 데이터가 충분히 많다면 아웃라이어 대체보다 제거가 편하고 효율적인 듯함.
## 
#### 최종 순위 및 느낀 점, 보완할 점
* 69 / 729
* 딥러닝을 사용하여 결승까지는 진출했으나, 역시나 최상위권은 LightGBM, XGBoost 등의 트리 기반 Gradient Boosting 방식 머신러닝이 차지함
  - 딥러닝의 미래는...?
* seed 값을 두 개 이상 사용하여 평균 살펴볼 것
* 하드웨어가 버텨준다면 StratifiedKFold 사용할 것
* 1등 참가자가 중요시한 Feature Engineering 에 대해 공부할 것
