import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

heart_data = pd.read_csv('./heart_disease_new.csv')

heart_data.insert(5, 'bmi', round((heart_data['weight']/(heart_data['height']/100)**2), 2))
heart_data = heart_data.drop(heart_data.query('ap_hi>220 or ap_lo>180 or ap_hi<40 or ap_lo<40').index, axis=0)
heart_data = heart_data.drop(heart_data.query('ap_hi<ap_lo').index, axis=0)

heart_data['age'] = round(heart_data['age']/365.25,2)
heart_data = heart_data.drop(heart_data.query('age<30 or age>60').index, axis=0)

X = heart_data.drop(columns=['cardio','id'],axis=1)
Y = heart_data['cardio']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,train_size=0.8, stratify=Y)

numeric=['age', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
sc=StandardScaler()
X_train[numeric]=sc.fit_transform(X_train[numeric])
X_test[numeric]=sc.transform(X_test[numeric])

model=LogisticRegression()
model.fit(X_train, Y_train)
pickle.dump(model, open('heart_new.pkl', 'wb'))

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data : ', test_data_accuracy)
