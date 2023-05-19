import joblib
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()

text = {}

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

LR = joblib.load('model_lr.joblib')
DT = joblib.load('model_dt.joblib')
GB = joblib.load('model_gb.joblib')
RF = joblib.load('model_rf.joblib')

merged_data_true_fake = pd.read_csv('merged.csv')
X_train = pd.read_csv('X_train.csv')

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
      X_train = pd.read_csv('X_train.csv')
      fitness = vectorization.fit_transform(X_train)
      new_xv_test = vectorization.transform(new_x_test)
      pred_LR = LR.predict(new_xv_test)
      pred_DT = DT.predict(new_xv_test)
      pred_GB = GB.predict(new_xv_test)
      pred_RF = RF.predict(new_xv_test)
      return f"LR Predict: {output_lable(pred_LR[0])} \nDT Predict: {output_lable(pred_DT[0])} \nGB Predict: {output_lable(pred_GB[0])} \nRF Predict: {output_lable(pred_RF[0])} "

  for i in range(len(merged_data_true_fake['text'])):
    # a = print('\n\n --- Ответ: ', merged_data_true_fake['class'][i], ' --- ')
    text['Ответ'] = merged_data_true_fake['class'][i]
    # b = print(merged_data_true_fake['text'][i])
    text['Ответ_ML'] = manual_testing(merged_data_true_fake['text'][i])

test_model()
def save_info(model_info):
  with open('result.txt', "w") as file:
      for key, value in text.items():
          file.write(f"{key}: {value}\n")
  print("Сохранение: result.txt")
