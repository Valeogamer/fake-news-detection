import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

models_info = {}

def save_info(model_info):
  with open('info.txt', "w") as file:
      for key, value in model_info.items():
          file.write(f"{key}: {value}\n")
  print("Сохранение: info.txt")

print("Загрузка необходимых данных")
# with open('xv_train.pickle', 'rb') as f:
#     xv_train = pickle.load(f)

# with open('xv_test.pickle', 'rb') as f:
#     xv_test = pickle.load(f)

Y_train = pd.read_csv('Y_train.csv')
Y_test = pd.read_csv('Y_test.csv')
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
merged_data_true_fake = pd.read_csv('merged.csv')
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(X_test)

print('-----Модели-----')
print("---LogisticRegression---")
import time
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, Y_train)
start = time.time()
pred_lr = LR.predict(xv_test)
result_time = time.time() - start
print(result_time)
LR.score(xv_test, Y_test)
print(classification_report(Y_test, pred_lr))
models_info['LR time:'] = result_time
models_info['LR CR: '] = classification_report(Y_test, pred_lr)
joblib.dump(LR, 'model_lr.joblib')

print("---DecisionTreeClassifier---")
from sklearn.tree import DecisionTreeClassifier
import time
DT = DecisionTreeClassifier()
DT.fit(xv_train, Y_train)
start = time.time()
pred_dt = DT.predict(xv_test)
result_time = time.time() - start
print(result_time)
DT.score(xv_test, Y_test)
print(classification_report(Y_test, pred_dt))
models_info['DT time:'] = result_time
models_info['DT CR: '] = classification_report(Y_test, pred_lr)
joblib.dump(DT, 'model_dt.joblib')

print("---GradientBoostingClassifier---")
from sklearn.ensemble import GradientBoostingClassifier
import time
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, Y_train)
start = time.time()
pred_gb = GB.predict(xv_test)
result_time =time.time() - start
print(result_time)
GB.score(xv_test, Y_test)
print(classification_report(Y_test, pred_gb))
models_info['GB time:'] = result_time
models_info['GB CR: '] = classification_report(Y_test, pred_lr)
joblib.dump(GB, 'model_gb.joblib')

print("---RandomForestClassifier---")
from sklearn.ensemble import RandomForestClassifier
import time
RF = RandomForestClassifier()
RF.fit(xv_train, Y_train)
start = time.time()
pred_rf = RF.predict(xv_test)
result_time = time.time() - start
print(result_time)
RF.score(xv_test, Y_test)
print(classification_report(Y_test, pred_rf))
models_info['RF time:'] = result_time
models_info['RF CR: '] = classification_report(Y_test, pred_lr)
joblib.dump(RF, 'model_rf.joblib')

save_info(models_info)
