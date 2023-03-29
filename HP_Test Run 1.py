import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

data = pd.read_csv("heart.csv")
x = data.iloc[:,:-1].values
y = data['target']
sc = StandardScaler()
sc.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# classifier = RandomForestClassifier(random_state=1)
# classifier.fit(x_train, y_train)
model = pickle.load(open("heart_disease_model.sav", "rb"))
pickle.dump(model, open('Heart.pkl', 'wb'))

y_pred = model.predict(x_test)
c_m = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("The confusion matrix is \n",c_m)
print("The accuracy score is \n",acc)