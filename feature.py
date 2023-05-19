import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()

def vectoriz(X_test, X_train):
  print('Векторизация Tidf')
  xv_train = vectorization.fit_transform(X_train)
  xv_test = vectorization.transform(X_test)
  return xv_train, xv_test

X_test = pd.read_csv('X_test.csv')
X_train = pd.read_csv('X_train.csv')
xv_train, xv_test = vectoriz(X_test, X_train)
with open('xv_train.pickle', 'wb') as f:
    pickle.dump(xv_train, f)
with open('xv_test.pickle', 'wb') as f:
    pickle.dump(xv_test, f)