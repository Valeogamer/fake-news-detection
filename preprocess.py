import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string

data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

def info_size(data_fake, data_true):
  print("Кол-во Fake: ", data_fake.shape[0], "\nКол-во True: ", data_true.shape[0])

def marks_class(data_fake, data_true):
  print("Присвоение отметок класcам: \n 1 - True \n 0 - False")
  data_fake['class'] = 0
  data_true['class'] = 1

def save_add_data(flag, data_fake_add, data_true_add = None):
  if flag == 'new':
    str_1 = 'Fake_new.csv'
    str_2 = 'True_new.csv'
    data_fake_add.to_csv(f'{str_1}', index = False)
    data_true_add.to_csv(f'{str_2}', index = False)
    print(f"Сохранение: {str_1}")
    print(f"Сохранение: {str_2}")
  elif flag == 'merge':
    data_fake_add.to_csv('merged.csv')
    print(f'Сохранение: merged.csv')
  elif flag == 'add':
    data_fake_add.to_csv(f'fake_add.csv', index = False)
    data_true_add.to_csv(f'true_add.csv', index = False)
    print(f"Сохранение: fake_add.csv")
    print(f"Сохранение: true_add.csv")

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

def spliter_learn(data, flag=None): 
  print("Разделение на тестовые и обучающие")
  x = data['text']
  y = data['class']
  X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)
  X_train.to_csv('X_train.csv')
  X_test.to_csv('X_test.csv')
  Y_train.to_csv('Y_train.csv')
  Y_test.to_csv('Y_test.csv')
  if flag:
    return X_train, X_test, Y_train, Y_test

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

info_size(data_fake, data_true)
marks_class(data_fake, data_true)

print('данные которые надо будет добавить при следуюущем обучении')
data_fake_add, data_true_add = split_data(data_fake, data_true)
save_add_data('add', data_fake_add, data_true_add)
info_size(data_fake, data_true)

print('Данные для тестов: 10 true, 10 fake')
data_fake_test, data_true_test = split_data(data_fake, data_true)
# ставим метки, объединяем, удаляем не нужные колонки и получаем 1 файл для теста
merged_data_true_fake = preproces_data(data_fake_test, data_true_test)
save_add_data('merge', merged_data_true_fake)
info_size(data_fake, data_true)

save_add_data('new', data_fake, data_true)

# предобработка основных данных
merged_data = preproces_data(data_fake, data_true)
# Удаление мусора
merged_data['text'] = merged_data['text'].apply(wordopt)
# Разделение на обучающую и тестовую
spliter_learn(merged_data)

