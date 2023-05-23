import pandas as pd
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer

models_info = {}
models_info_new = {}

def save_info(model_info, new):
    with open(f'info_{new}.txt', "w") as file:
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
  models_info[f'accuracy {new}:'] = LR.score(X_test, y_test)
  models_info[f'LR CR {new}:'] = classification_report(y_test, pred_lr)
  save_info(models_info, new)

  with open(f'model_lr_{new}.pickle', 'wb') as file:
    pickle.dump(LR, file)

models(X_train, y_train, X_test, y_test, new=old)
models(X_train_new, y_train_new, X_test_new, y_test_new, new=new)
