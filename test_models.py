import pandas as pd
import string
import re
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()

LR = joblib.load('model_lr.joblib')
DT = joblib.load('model_dt.joblib')
GB = joblib.load('model_gb.joblib')
RF = joblib.load('model_rf.joblib')
  
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

data_f = pd.read_csv('Fake_new.csv')
data_t = pd.read_csv('True_new.csv')

data = preproces_data(data_f, data_t)
data['text'] = data['text'].apply(wordopt)
x = data['text']
y = data['class']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25)


xv_train = vectorization.fit_transform(X_train)
# xv_test = vectorization.transform(X_test)
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
