import zipfile
import pandas as pd
with zipfile.ZipFile('csv-files.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')
print(data_fake.head())
