from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
main_data = pd.read_csv('main_data.csv')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(main_data['text'])
with open('vectorizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('features.pickle', 'wb') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
