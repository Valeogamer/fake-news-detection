import pandas as pd
data_fake = pd.read_csv('/home/runner/work/Test.csv')
a = data_fake.head()
print(a)
a.to_csv('Tester.csv')
