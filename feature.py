from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
main_data = pd.read_csv('main_data.csv')
main_data_new = pd.read_csv('main_data_new.csv')
def vec(main_data, name):
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(main_data['text'])
  with open(f'vectorizer_{name}.pickle', 'wb') as handle:
      pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
  with open(f'features_{name}.pickle', 'wb') as handle:
      pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
old = 'old'
new = 'new'
vec(main_data, old)
vec(main_data_new, new)
