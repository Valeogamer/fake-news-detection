import zipfile
import pandas as pd
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
data_fake = pd.read_csv('data/Fake.csv')
data_true = pd.read_csv('data/True.csv')
print(data_fake.head())
