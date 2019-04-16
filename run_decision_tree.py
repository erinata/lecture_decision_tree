import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("dataset.csv")

print(dataset.head())

target = dataset.iloc[:,30].values
print(target[1:40])

data = dataset.iloc[:,0:30]
print(data.head())

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.2, random_state=1)

print("data_training")
print(data_training.head())
print("data_test")
print(data_test.head())




