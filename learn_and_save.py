import pandas as pd
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer

models_info = {}
models_info_new = {}

def save_info(model_info, new):
    with open(f'info_{new}.txt', "a") as file:
        for key, value in model_info.items():
            file.write(f"{key}: {value}\n")
    print(f"Сохранение: info_{new}.txt")



combined_df = pd.read_csv('main_data.csv')
with open('vectorizer_old.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)
with open('features_old.pickle', 'rb') as handle:
    X = pickle.load(handle)

X_train, X_test, y_train, y_test = train_test_split(X, combined_df['class'], test_size=0.2, random_state=42)

combined_df_new = pd.read_csv('main_data_new.csv')
with open('vectorizer_new.pickle', 'rb') as handle:
    vectorizer_new = pickle.load(handle)
with open('features_new.pickle', 'rb') as handle:
    X_new = pickle.load(handle)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, combined_df_new['class'], test_size=0.3, random_state=42)

print('-----Модели-----')
def models(X_train, y_train, X_test, y_test, new=None):
  print("---LogisticRegression---")
  LR = LogisticRegression()
  LR.fit(X_train, y_train)
  start = time.time()
  pred_lr = LR.predict(X_test)
  result_time = time.time() - start
  print(result_time)
  print(f'accuracy {new}:', LR.score(X_test, y_test))
  print(classification_report(y_test, pred_lr))
  models_info[f'LR time {new}:'] = result_time
  models_info[f'accuracy LR {new}:'] = LR.score(X_test, y_test)
  models_info[f'LR CR {new}:'] = classification_report(y_test, pred_lr)
  save_info(models_info, new)
  
  print("---DecisionTreeClassifier---")
  DT = DecisionTreeClassifier()
  DT.fit(X_train, y_train)
  start = time.time()
  pred_dt = DT.predict(X_test)
  result_time = time.time() - start
  print(result_time)
  print(f'accuracy {new}:', DT.score(X_test, y_test))
  print(classification_report(y_test, pred_dt))
  models_info['DT time:'] = result_time
  models_info[f'accuracy DT {new}:'] = DT.score(X_test, y_test)
  models_info['DT CR: '] = classification_report(y_test, pred_lr)
  save_info(models_info, new)
  
  print("---GradientBoostingClassifier---")
  GB = GradientBoostingClassifier(random_state=0)
  GB.fit(X_train, y_train)
  start = time.time()
  pred_gb = GB.predict(X_test)
  result_time =time.time() - start
  print(result_time)
  print(f'accuracy {new}:', GB.score(X_test, y_test))
  print(classification_report(y_test, pred_gb))
  models_info['GB time:'] = result_time
  models_info[f'accuracy GB {new}:'] = GB.score(X_test, y_test)
  models_info['GB CR: '] = classification_report(y_test, pred_lr)
  save_info(models_info, new)

  print("---RandomForestClassifier---")
  RF = RandomForestClassifier()
  RF.fit(X_train, y_train)
  start = time.time()
  pred_rf = RF.predict(X_test)
  result_time = time.time() - start
  print(result_time)
  print(f'accuracy {new}:', RF.score(X_test, y_test))
  print(classification_report(y_test, pred_rf))
  models_info['RF time:'] = result_time
  models_info[f'accuracy RF {new}:'] = RF.score(X_test, y_test)
  models_info['RF CR: '] = classification_report(y_test, pred_lr)
  save_info(models_info, new)

  with open(f'model_lr_{new}.pickle', 'wb') as file:
    pickle.dump(LR, file)
  with open(f'model_dt_{new}.pickle', 'wb') as file:
    pickle.dump(DT, file)
  with open(f'model_gb_{new}.pickle', 'wb') as file:
    pickle.dump(GB, file)
  with open(f'model_rf_{new}.pickle', 'wb') as file:
    pickle.dump(RF, file)
old = 'old'
new = 'new'
models(X_train, y_train, X_test, y_test, new=old)
models(X_train_new, y_train_new, X_test_new, y_test_new, new=new)
