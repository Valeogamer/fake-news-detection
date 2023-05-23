import pandas as pd
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
    data_fake_add.to_csv(f'{str_1}', index = False, header=True)
    data_true_add.to_csv(f'{str_2}', index = False, header=True)
    print(f"Сохранение: {str_1}")
    print(f"Сохранение: {str_2}")
  elif flag == 'merge':
    data_fake_add.to_csv('merged.csv', index=False,  header=True)
    print(f'Сохранение: merged.csv')
  elif flag == 'add':
    data_fake_add.to_csv(f'fake_add.csv', index = False,  header=True)
    data_true_add.to_csv(f'true_add.csv', index = False,  header=True)
    print(f"Сохранение: fake_add.csv")
    print(f"Сохранение: true_add.csv")

def split_data(data_fake, data_true, n): # вернет два dataframe. в каждом по 10 элеметов из true и из false
  print("Подготовка данных для тестов")
  print("По 10 данных с каждого файла для тестов!")
  data_fake_testing = data_fake.tail(n)
  for i in range(data_fake.shape[0]-1, (data_fake.shape[0] - n), -1):
    data_fake.drop([i], axis = 0, inplace = True)

  data_true_testing = data_true.tail(n)
  for i in range(data_true.shape[0]-1, (data_true.shape[0] - n), -1):
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
data_fake_add, data_true_add = split_data(data_fake, data_true, n=10000)
save_add_data('add', data_fake_add, data_true_add)
info_size(data_fake, data_true)

print('Данные для тестов: 10 true, 10 fake')
data_fake_test, data_true_test = split_data(data_fake, data_true, n=10)
# ставим метки, объединяем, удаляем не нужные колонки и получаем 1 файл для теста
merged_data_true_fake = preproces_data(data_fake_test, data_true_test)
save_add_data('merge', merged_data_true_fake)
info_size(data_fake, data_true)

save_add_data('new', data_fake, data_true)

data_fake_old = pd.read_csv('Fake_new.csv')
data_true_old = pd.read_csv('True_new.csv')
# предобработка основных данных
merged_data = preproces_data(data_fake_old, data_true_old)
# Удаление мусора
merged_data['text'] = merged_data['text'].apply(wordopt)
# Разделение на обучающую и тестовую
merged_data.to_csv('main_data.csv', index=False, header=True)
info_size(data_fake, data_true)
def compound():
  print("Объединение")
  data_fake_orig = pd.read_csv('Fake_new.csv')
  data_fake_add = pd.read_csv('fake_add.csv')
  fake = pd.concat([data_fake_orig, data_fake_add], axis=0)
  fake.to_csv('Fake_up_learn.csv', index=False)
  data_true_orig = pd.read_csv('True_new.csv')
  data_true_add = pd.read_csv('true_add.csv')
  true = pd.concat([data_fake_orig, data_fake_add], axis=0)
  true.to_csv('True_up_learn.csv', index=False)
compound()
data_fake_new = pd.read_csv('Fake_up_learn.csv')
data_true_new = pd.read_csv('True_up_learn.csv')
# предобработка основных данных
merged_data_new = preproces_data(data_fake_new, data_true_new)
# Удаление мусора
merged_data_new['text'] = merged_data_new['text'].apply(wordopt)
# Разделение на обучающую и тестовую
merged_data.to_csv('main_data_new.csv', index=False, header=True)
print(merged_data.shape)
print(merged_data_new.shape)
