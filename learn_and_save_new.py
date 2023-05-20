import pandas as pd
import string
import re
from sklearn.metrics import classification_report
# import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()

models_info = {}

def save_info(model_info):
  with open('new_info.txt', "w") as file:
      for key, value in model_info.items():
          file.write(f"{key}: {value}\n")
  print("Сохранение: new_info.txt")
  
def preproces_data(data_fake, data_true):
  print("Предобработка!")
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
  text = text.lower()
  text = re.sub('\[.*?]', '', text)
  text = re.sub("\\W", " ", text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  return text

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

compound()

data_f = pd.read_csv('Fake.csv')
data_t = pd.read_csv('True.csv')

data = preproces_data(data_f, data_t)
data['text'] = data['text'].apply(wordopt)
x = data['text']
y = data['class']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25)


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
# joblib.dump(LR, 'model_lr.joblib')

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
# joblib.dump(DT, 'model_dt.joblib')

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
# joblib.dump(GB, 'model_gb.joblib')

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
# joblib.dump(RF, 'model_rf.joblib')

save_info(models_info)

merged_data_true_fake = pd.read_csv('merged.csv')

text = {}
def test_model():
  print("Tecтирование модели!")
  print("Внимание! Функция ввода временно отключена!")
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
      new_xv_test = vectorization.transform(new_x_test)
      pred_LR = LR.predict(new_xv_test)
      pred_DT = DT.predict(new_xv_test)
      pred_GB = GB.predict(new_xv_test)
      pred_RF = RF.predict(new_xv_test)
      return f"LR Predict: {output_lable(pred_LR[0])} \nDT Predict: {output_lable(pred_DT[0])} \nGB Predict: {output_lable(pred_GB[0])} \nRF Predict: {output_lable(pred_RF[0])} "

  for i in range(len(merged_data_true_fake['text'])):
    # a = print('\n\n --- Ответ: ', merged_data_true_fake['class'][i], ' --- ')
    text[f'{i}.Ответ:'] = merged_data_true_fake['class'][i]
    text[f'{i}.Text:'] = merged_data_true_fake['text'][i]
    text[f'{i}.Ответ_ML:'] = manual_testing(merged_data_true_fake['text'][i])

test_model()
def save_info(text):
  with open('result.txt', "w") as file:
      for key, value in text.items():
          file.write(f"{key}: {value}\n")
  print("Сохранение: result.txt")

save_info(text)
