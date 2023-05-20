import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
X_train = pd.read_csv('X_train.csv')
xv_train = vectorization.fit_transform(X_train)
joblib.dump(vectorization, 'vectorizer.joblib')
