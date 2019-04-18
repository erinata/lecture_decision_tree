import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("dataset.csv")
target = dataset.iloc[:,30].values
# print(target[1:40])

data = dataset.iloc[:,0:30]
# print(data.head())

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.2, random_state=1)

# print("data_training")
# print(data_training.head())
# print("data_test")
# print(data_test.head())
