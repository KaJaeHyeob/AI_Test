import numpy as np
import pandas as pd
import tensorflow as tf
import os
import math
import keras
import tensorflow.keras.backend as K

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import adam

## 시드 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

## CWD 확인
print(os.getcwd())

## 데이터 읽기
train = pd.read_csv('dataset/train.csv', index_col='id')
test = pd.read_csv('dataset/test.csv', index_col='id')
sample = pd.read_csv('dataset/sample_submission.csv')

## replace(), rename() 사용을 위한 사전 만들기
## 'QSO': 0, 'STAR_RED_DWARF': 1, 'SERENDIPITY_BLUE': 2, 'STAR_BHB': 3, ...
## 0: 'QSO', 1: 'STAR_RED_DWARF', 2: 'SERENDIPITY_BLUE', 3: 'STAR_BHB', ...
unique_labels = train['type'].unique()
label_dict = {val: i for i, val in enumerate(unique_labels)}
i2lb = {v: k for k, v in label_dict.items()}

## type 따로 저장 및 제거, 테스트셋 in 따로 저장
labels = train['type']
train = train.drop(columns=['fiberID', 'type'])
test = test.drop(columns=['fiberID'])
test_ids = test.index
test_ids = pd.DataFrame(test_ids, columns=['id'], index=test.index)

## 학습셋 테스트셋 병합
all_data = pd.concat([train, test])
all_ids = pd.DataFrame(all_data.index, columns=["id"])

## 10이하 30이상 아웃라이어 10, 30으로 대체
for i in all_data.index:
    for col in all_data.columns:
        out_val = all_data.loc[i][col]
        if out_val < 10:
            all_data.loc[i][col] = 10
        elif out_val > 30:
            all_data.loc[i][col] = 30

all_data_save = pd.concat([all_ids, all_data], axis=1)
all_data_save.to_csv("dataset/all_data.csv", index=False)

'''
## IQR 함수 정의
def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 10)
    upper_bound = q3 + (iqr * 10)
    return np.where((data > upper_bound) | (data < lower_bound))


## IQR 사용해서 아웃라이어 찾기
out_i = {i for i in outliers_iqr(train['psfMag_u'])[0]}

## iloc : 행번호, loc : 인덱스
print(train.loc[out_i])

## 아웃라이어 제거
train = train.drop(index=out_i)
'''

'''
## LOF 사용 아웃라이어 제거 작업
lof_model = LocalOutlierFactor(n_neighbors=500)
pred_outliers = lof_model.fit_predict(train)
pred_outliers = pd.DataFrame(pred_outliers)
pred_outliers = pred_outliers.rename(columns={0:"out"})
train = pd.concat([train, pred_outliers], 1)
out_index = train[train["out"]==-1].index
print(out_index.size)
train = train.drop(out_index)
train = train.drop(columns="out")
labels = labels.drop(out_index)
print(labels.value_counts())
train.to_csv("dataset/train_out.csv")
'''

'''
## 음수 아웃라이어 제거 작업
print((train < 0).any(1))
print(train[(train < 0).any(1)])
minus_outliers = pd.DataFrame((train < 0).any(1))
minus_outliers = minus_outliers.rename(columns={0:"out"})
train = pd.concat([train, minus_outliers], 1)
out_index = train[train["out"]==True].index
print(out_index.size)
train = train.drop(out_index)
train = train.drop(columns="out")
labels = labels.drop(out_index)
print(labels.value_counts())
'''

'''
## IF 사용 아웃라이어 제거 작업
if_model = IsolationForest(max_samples=300000, random_state=1)
if_model.fit(all_data)
pred_outliers = if_model.predict(train)
pred_outliers = pd.DataFrame(pred_outliers, index=train.index)
print((pred_outliers==-1).any(1))
print(pred_outliers[(pred_outliers==-1).any(1)])
print(pred_outliers[(pred_outliers==-1).any(1)].size)
pred_outliers = pred_outliers.rename(columns={0:"out"})
train = pd.concat([train, pred_outliers], 1)
out_index = train[train["out"]==-1].index
train = train.drop(out_index)
train = train.drop(columns="out")
labels = labels.drop(out_index)
print(labels.value_counts())
'''

'''
## 변수별 평균 구하기
mean_x = train.mean()
mean_x = pd.DataFrame(mean_x)
mean_x = np.transpose(mean_x)
mean_x
'''

## 정규화
## 이상치 영향 최소화 위해서 RobustScaler() 사용
scaler = RobustScaler()
## scaler_2 = StandardScaler()
## scaler_3 = MinMaxScaler()
all_data_scale = scaler.fit_transform(all_data)
all_data = pd.DataFrame(all_data_scale, columns=all_data.columns, index=all_data.index)

## 학습셋 테스트셋 분할
train = all_data.loc[0:199990]
test = all_data.loc[199991:209999]

## X, Y 나누기
train_x = train
train_y = labels.replace(label_dict)

'''
## 원-핫 인코딩
## 오차 함수로 sparse_categorical_crossentropy가 새로 나오면서 원-핫 인코딩 필요가 없어짐
train_y = np_utils.to_categorical(train_y)
'''

## X, Y 학습셋 안에서 학습셋 테스트셋 구분하기
## x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=seed)


def mish(x):
    return x * K.tanh(K.softplus(x))


## 모델 설정
model = Sequential()
model.add(Dense(500, input_dim=len(train_x.columns), activation=mish))
model.add(Dropout(0.1))
model.add(Dense(400, activation=mish))
model.add(Dropout(0.1))
model.add(Dense(300, activation=mish))
model.add(Dropout(0.1))
model.add(Dense(200, activation=mish))
model.add(Dropout(0.1))
model.add(Dense(100, activation=mish))
model.add(Dropout(0.1))
model.add(Dense(50, activation=mish))
model.add(Dense(19, activation='softmax'))

## 경사 하강법 설정
opt = adam(lr=0.001)

## 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

## 모델 자동 저장 설정
model_path = "model/{epoch:04d}-{val_loss:.4f}.h5"
checkpointer = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)

## 모델 실행
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=10000, validation_split=0.2,
          epochs=9999, verbose=2, callbacks=[early_stopping_callback, checkpointer])

## 모델 저장
model.save('model/0241-0.3586.h5')

## 모델 삭제
del model

## 모델 불러오기
model = load_model('submission/0234-0.3492.h5', custom_objects={"mish":mish})

## 예측 실행
pred_mat = model.predict(test)

## 제출 파일 내보내기
submission = pd.DataFrame(pred_mat, index=test.index)
submission = submission.rename(columns=i2lb)
submission = submission[sample.columns]
submission = pd.concat([test_ids, submission], axis=1)
submission.to_csv("submission/0241-0.3586.csv", index=False)


###########################################################################################
###########################################################################################


## 행, 열 조작 참고
## https://blog.naver.com/PostView.nhn?blogId=rising_n_falling&logNo=221629326893

## 딮러닝 기초 내용
## https://nmhkahn.github.io/NN

## 활성화 함수 정리
## https://www.simonwenkel.com/2018/05/15/activation-functions-for-neural-networks.html
