import pandas as pd
data_fake = pd.read_csv('Test.csv')
a = data_fake.head()
print(a)
a.to_csv('Tester.csv')
