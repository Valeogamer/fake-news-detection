import pandas as pd
import pickle
import re
import string
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

models_info = {}

def save_info(model_info):
  with open('info.txt', "w") as file:
      for key, value in model_info.items():
          file.write(f"{key}: {value}\n")
  print("Сохранение: info.txt")

def compound():
  print("Объединение")
  data_fake_orig = pd.read_csv('Fake_new.csv')
  data_fake_add = pd.read_csv('fake_add.csv')
  fake = pd.concat([data_fake_orig, data_fake_add], axis=0)
  fake.to_csv('Fake.csv', index=False)
  data_true_orig = pd.read_csv('True_new.csv')
  data_true_add = pd.read_csv('true_add.csv')
  true = pd.concat([data_true_orig, data_true_add], axis=0)
  true.to_csv('True.csv', index=False)

def spliter_learn(data, test_size=0.25): 
  print("Разделение на тестовые и обучающие")
  x = data['text']
  y = data['class']
  X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size)
  return X_train, X_test, Y_train, Y_test

def preproces_data(data_fake, data_true):
  print("Предобработка!")
  data_fake['class'] = 0
  data_true['class'] = 1
  print("Объединение данных")
  data_merge = pd.concat([data_fake, data_true], axis = 0)
  print("Удаление не нужных колонок")
  data_merge = data_merge.drop(['title', 'subject', 'date'], axis = 1)
  print("Смешивание данных")
  data_merge = data_merge.sample(frac = 1) # cмешивание данных
  print("Переиндексация")
  data_merge.reset_index(inplace = True)
  data_merge.drop(['index'], axis=1, inplace=True)
  return data_merge

def wordopt(text):
  """
    Удаление с текста всякого мусора
  """
  text = text.lower()
  text = re.sub('\[.*?]', '', text)
  text = re.sub("\\W", " ", text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  return text

print("Загрузка необходимых данных")
with open('xv_train.pickle', 'rb') as f:
    xv_train = pickle.load(f)

with open('xv_test.pickle', 'rb') as f:
    xv_test = pickle.load(f)

data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# предобработка основных данных
merged_data = preproces_data(data_fake, data_true)
# Удаление мусора
merged_data['text'] = merged_data['text'].apply(wordopt)
# Разделение на обучающую и тестовую
X_train, X_test, Y_train, Y_test = spliter_learn(merged_data, test_size=0.40)

merged_data_true_fake = pd.read_csv('merged.csv')

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

save_info(models_info)
