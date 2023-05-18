import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
import os
import zipfile
with zipfile.ZipFile('csv-files.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
data_fake = pd.read_csv('/data/Fake.csv')
data_true = pd.read_csv('/data/True.csv')
def info_size(data_fake, data_true):
  print("Кол-во Fake: ", data_fake.shape[0], "\nКол-во True: ", data_true.shape[0])
def marks_class(data_fake, data_true):
  print("Присвоение отметок класcам: \n 1 - True \n 0 - False")
  data_fake['class'] = 0
  data_true['class'] = 1
info_size(data_fake, data_true)
marks_class(data_fake, data_true)

def split_data(data_fake, data_true): # вернет два dataframe. в каждом по 10 элеметов из true и из false
  print("Подготовка данных для тестов")
  print("По 10 данных с каждого файла для тестов!")
  data_fake_testing = data_fake.tail(10)
  for i in range(data_fake.shape[0]-1, (data_fake.shape[0] - 10), -1):
    data_fake.drop([i], axis = 0, inplace = True)

  data_true_testing = data_true.tail(10)
  for i in range(data_true.shape[0]-1, (data_true.shape[0] - 10), -1):
    data_true.drop([i], axis = 0, inplace = True)
  
  return data_fake_testing, data_true_testing

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

def compound():
  print("Объединение")
  data_fake_orig = pd.read_csv('file1.csv')
  data_fake_add = pd.read_csv('file2.csv')
  fake = pd.concat([data_fake_orig, data_fake_add], axis=0)
  fake.to_csv('Fake.csv', index=False)
  data_true_orig = pd.read_csv('file1.csv')
  data_true_add = pd.read_csv('file2.csv')
  true = pd.concat([data_fake_orig, data_fake_add], axis=0)
  true.to_csv('True.csv', index=False)

def save_add_data(data_fake_add, data_true_add):
  print("Сохранение")
  data_fake_add.to_csv('Fake_add.csv', index = False)
  data_true_add.to_csv('True_add.csv', index = False)

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

def spliter_learn(data, test_size=0.25): 
  print("Разделение на тестовые и обучающие")
  x = data['text']
  y = data['class']
  X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25)
  X_train.to_csv('X_train.csv')
  return X_train, X_test, Y_train, Y_test

def vectoriz(X_test, X_train):
  print('Векторизация Tidf')
  xv_train = vectorization.fit_transform(X_train)
  xv_test = vectorization.transform(X_test)
  return xv_train, xv_test

def save_info(model_info):
  with open('info.txt', "w") as file:
      for key, value in model_info.items():
          file.write(f"{key}: {value}\n")

# данные которые надо будет добавить при следуюущем обучении
data_fake_add, data_true_add = split_data(data_fake, data_true)
save_add_data(data_fake_add, data_true_add)
info_size(data_fake, data_true)

# для тестирования
# получаем 10 true, 10 fake
data_fake_test, data_true_test = split_data(data_fake, data_true)
# ставим метки, объединяем, удаляем не нужные колонки и получаем 1 файл для теста
merged_data_true_fake = preproces_data(data_fake_test, data_true_test)
info_size(data_fake, data_true)

# предобработка основных данных
merged_data = preproces_data(data_fake, data_true)
# Удаление мусора
merged_data['text'] = merged_data['text'].apply(wordopt)
# Разделение на обучающую и тестовую
X_train, X_test, Y_train, Y_test = spliter_learn(merged_data, test_size=0.20)
# Векторизация 
xv_train, xv_test = vectoriz(X_test, X_train)
info_size(data_fake, data_true)

print('-----Модели-----')
models_info = {}
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

def test_model():
  print("Tecтирование модели!")
  print("Внимание! Функция ввода временно отключена!")
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorization = TfidfVectorizer()
  def output_lable(n):
    if n == 0:
      return "Лживая новость!"
    elif n == 1:
      return "Правдивая новость"
    else:
      return 'Необработанный случай'

  def manual_testing(news):
      testing_news = {"text": [news]}
      new_def_test = pd.DataFrame(testing_news)
      new_def_test["text"] = new_def_test["text"].apply(wordopt)
      new_x_test = new_def_test["text"]
      X_train = pd.read_csv('X_train.csv')
      fitness = vectorization.fit_transform(X_train)
      new_xv_test = vectorization.transform(new_x_test)
      pred_LR = LR.predict(new_xv_test)
      pred_DT = DT.predict(new_xv_test)
      pred_GB = GB.predict(new_xv_test)
      pred_RF = RF.predict(new_xv_test)
      return print(f"LR Predict: {output_lable(pred_LR[0])} \nDT Predict: {output_lable(pred_DT[0])} \nGB Predict: {output_lable(pred_GB[0])} \nRF Predict: {output_lable(pred_RF[0])} ")

  for i in range(len(merged_data_true_fake['text'])):
    print('\n\n --- Ответ: ', merged_data_true_fake['class'][i], ' --- ')
    print(merged_data_true_fake['text'][i])
    manual_testing(merged_data_true_fake['text'][i])

test_model()
