import zipfile
import pandas as pd
# with zipfile.ZipFile('csv-files.zip', 'r') as zip_ref:
#     zip_ref.extractall('data')
data_fake = pd.read_csv('/home/runner/work/fake-news-detection/fake-news-detection/Fake.csv')
data_true = pd.read_csv('/home/runner/work/fake-news-detection/fake-news-detection/True.csv')
print(data_fake.head())
print(data_true.head())
a = data_fake.head()
a.to_csv('Test.csv')
