import pandas as pd
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

models_info = {}

def save_info(model_info):
    with open('info.txt', "w") as file:
        for key, value in model_info.items():
            file.write(f"{key}: {value}\n")
    print("Сохранение: info.txt")

# Загрузка объединенного датафрейма
combined_df = pd.read_csv('main_data.csv')

# Загрузка векторизатора и матрицы признаков
with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

with open('features.pickle', 'rb') as handle:
    X = pickle.load(handle)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, combined_df['class'], test_size=0.2, random_state=42)

print('-----Модели-----')
print("---LogisticRegression---")
import time
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
start = time.time()
pred_lr = LR.predict(X_test)
result_time = time.time() - start
print(result_time)
print('accuracy:', LR.score(X_test, y_test))
print(classification_report(y_test, pred_lr))
models_info['LR time:'] = result_time
models_info['accuracy:'] = LR.score(X_test, y_test)
models_info['LR CR:'] = classification_report(y_test, pred_lr)
save_info(models_info)
