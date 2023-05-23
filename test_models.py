import pandas as pd
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

def save_info(text):
    with open('result.txt', "w") as file:
        for key, value in text.items():
            file.write(f"{key}: {value}\n")
    print("Сохранение: result.txt")


def test_model():
    print("Тестирование модели!")
    print("Внимание! Функция ввода временно отключена!")

    def output_label(n):
        if n == 0:
            return "Лживая новость!"
        elif n == 1:
            return "Правдивая новость"
        else:
            return 'Необработанный случай'

    def manual_testing(news):
        # Векторизация текста
        X_new = vectorizer.transform([news])

        # Предсказание метки
        pred_LR = LR.predict(X_new)

        return f"LR Predict: {output_label(pred_LR[0])}"

    # Загрузка объединенного датафрейма
    merged_data_true_fake = pd.read_csv('merged.csv')

    text = {}
    for i in range(len(merged_data_true_fake['text'])):
        text[f'{i}.Ответ:'] = merged_data_true_fake['class'][i]
        text[f'{i}.Text:'] = merged_data_true_fake['text'][i]
        text[f'{i}.Ответ_ML:'] = manual_testing(merged_data_true_fake['text'][i])

    save_info(text)


# Загрузка модели
with open('model_lr.pickle', 'rb') as handle:
    LR = pickle.load(handle)

# Загрузка векторизатора
with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

test_model()
