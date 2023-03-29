import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import numpy as np

data = pd.read_csv("diabetes (1).csv")

x = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=0)
print(train_x)
sc = StandardScaler()
train_x = sc.fit_transform(train_x) 
test_x = sc.fit_transform(test_x)
# scaler = StandardScaler()

# cls = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=1)
# cls.fit(train_x, train_y)
# pickle.dump(cls, open("Diabetes.pkl", "wb"))

model = pickle.load(open("diabetes_model.sav", "rb"))
input_data = (11,143,94,33,146,36.6,0.254,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = sc.transform(input_data_reshaped)
print(std_data)

prediction = model.predict(std_data)
print(prediction)
# print(prediction)
if (prediction == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

