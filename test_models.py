import pandas as pd
import pickle

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
        X_new = vectorizer.transform([news])
        pred_LR = LR.predict(X_new)
        pred_DT = DT.predict(X_new)
        pred_GB = GB.predict(X_new)
        pred_RF = RF.predict(X_new)
        return print(f"LR Predict: {output_lable(pred_LR[0])} \nDT Predict: {output_lable(pred_DT[0])} \nGB Predict: {output_lable(pred_GB[0])} \nRF Predict: {output_lable(pred_RF[0])} ")


    merged_data_true_fake = pd.read_csv('merged.csv')
    text = {}
    for i in range(len(merged_data_true_fake['text'])):
        text[f'{i}.Ответ:'] = merged_data_true_fake['class'][i]
        text[f'{i}.Text:'] = merged_data_true_fake['text'][i]
        text[f'{i}.Ответ_ML:'] = manual_testing(merged_data_true_fake['text'][i])

    save_info(text)

with open('model_lr_old.pickle', 'rb') as handle:
    LR = pickle.load(handle)
with open('model_dt_old.pickle', 'rb') as handle:
    DT = pickle.load(handle)
with open('model_gb_old.pickle', 'rb') as handle:
    GB = pickle.load(handle)
with open('model_rf_old.pickle', 'rb') as handle:
    RF = pickle.load(handle)
with open('vectorizer_old.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)
test_model()

def save_info_new(text):
    with open('result_new.txt', "w") as file:
        for key, value in text.items():
            file.write(f"{key}: {value}\n")
    print("Сохранение: result_new.txt")


def test_model_new():
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
        X_new = vectorizer_new.transform([news])
        pred_LR = LR.predict(X_new)
        pred_DT = DT.predict(X_new)
        pred_GB = GB.predict(X_new)
        pred_RF = RF.predict(X_new)
        return print(f"LR Predict: {output_lable(pred_LR[0])} \nDT Predict: {output_lable(pred_DT[0])} \nGB Predict: {output_lable(pred_GB[0])} \nRF Predict: {output_lable(pred_RF[0])} ")

    merged_data_true_fake = pd.read_csv('merged.csv')
    text = {}
    for i in range(len(merged_data_true_fake['text'])):
        text[f'{i}.Ответ:'] = merged_data_true_fake['class'][i]
        text[f'{i}.Text:'] = merged_data_true_fake['text'][i]
        text[f'{i}.Ответ_ML:'] = manual_testing(merged_data_true_fake['text'][i])

    save_info_new(text)

with open('model_lr_new.pickle', 'rb') as handle:
    LR = pickle.load(handle)
with open('model_dt_new.pickle', 'rb') as handle:
    DT = pickle.load(handle)
with open('model_gb_new.pickle', 'rb') as handle:
    GB = pickle.load(handle)
with open('model_rf_new.pickle', 'rb') as handle:
    RF = pickle.load(handle)
with open('vectorizer_new.pickle', 'rb') as handle:
    vectorizer_new = pickle.load(handle)
test_model_new()
