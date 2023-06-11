from flask import Flask, render_template, request, redirect, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import numpy as np
import mysql.connector
import sqlite3
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
conn = mysql.connector.connect(host="localhost", user="root", password="Yash@2106", database="miniproj")
# conn = sqlite3.connect('database.db', check_same_thread=False)
cursor = conn.cursor()
modeld = pickle.load(open("diabetes_model.sav", "rb"))
modelh = pickle.load(open("heart_disease_model.sav", "rb"))
# modelhn = pickle.load(open("heart_new.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/diabetesm')
def diabetesm():
    return render_template('diabetes_med.html')


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/heartm')
def heartm():
    return render_template('heart_med.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/profile')
def profile():
    if 'id' in session:
        cursor.execute("""SELECT* FROM `USERS` WHERE `Username` LIKE '{}' """.format(session['id']))
        USERS = cursor.fetchone()
        return render_template('profile.html', USERS=USERS)
    else:
        return redirect('/login')

@app.route('/update')
def update():
    return render_template('update.html')


@app.route('/registration', methods=['POST'])
def registration():
    fullname = request.form.get('rname')
    username = request.form.get('rusername')
    email = request.form.get('remail')
    password = request.form.get('rpassword')
    age = request.form.get('rage')
    gender = request.form.get('rgender')

    # cursor.execute("""CREATE TABLE `USERS` (`Full Name` varchar(255), `Username` varchar(255), `Password` varchar(255), `Email` varchar(255), `Age` int, `Gender` varchar(255))""")

    cursor.execute("""INSERT INTO `USERS` (`Full Name`, `Username`, `Password`, `Email`, `Age`, `Gender`)
                        VALUES ('{}', '{}', '{}', '{}', '{}', '{}')"""
                   .format(fullname, username, password, email, age, gender))
    conn.commit()
    return redirect('/login')


@app.route('/login_validation', methods=['POST'])
def login_validation():
    username = request.form.get('username')
    password = request.form.get('password')

    cursor.execute("""SELECT* FROM `USERS` WHERE `Username` LIKE '{}' AND `Password` LIKE '{}'"""
                   .format(username, password))
    users = cursor.fetchall()
    if len(users) > 0:
        session['id'] = users[0][1]
        return redirect('/profile')
    else:
        return redirect('/login')


@app.route('/updating', methods=['GET', 'POST'])
def updating():
    if 'id' in session:
        fullname = request.form.get('uname')
        email = request.form.get('uemail')
        password = request.form.get('upassword')
        age = request.form.get('uage')
        gender = request.form.get('ugender')
        username = request.form.get('uusername')

        cursor.execute("""UPDATE `USERS` SET `Full Name` = '{}', `Password` = '{}', `Email` = '{}', `Age` = '{}', 
                `Gender` = '{}' WHERE `Username` LIKE '{}'""".format(fullname, password, email, age, gender, username))
        conn.commit()
    return redirect('/login')


@app.route('/logout')
def logout():
    session.pop('id')
    return redirect('/')


@app.route('/predictd', methods=['POST'])
def predictd():
    # For rendering results on the HTML page
    prg = request.form['prg']
    glc = request.form['gl']
    bp = request.form['bp']
    skt = request.form['sk']
    ins = request.form['ins']
    # bmi = request.form['BMI']
    dpf = request.form['ped']
    age = request.form['age']
    height = request.form['height']
    weight = request.form['weight']

    height = float(height) / 100
    weight = float(weight)

    bmi = weight / (height * height)

    prg = int(prg)
    glc = int(glc)
    bp = int(bp)
    skt = int(skt)
    ins = int(ins)
    dpf = float(dpf)
    age = int(age)
    sc = StandardScaler()

    # row_df = pd.DataFrame([pd.Series([prg, glc, bp, skt, ins, bmi, dpf, age])])
    # print(row_df)
    # prediction = modeld.predict_proba(row_df)[:, 1]
    data = np.array([[prg, glc, bp, skt, ins, bmi, dpf, age]])
    # my_prediction = modeld.predict(data)
    modeld = pickle.load(open("diabetes_model.sav", "rb"))
    input_data = (prg, glc, bp, skt, ins, bmi, dpf, age)

# changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
    std_data = sc.fit_transform(input_data_reshaped)
    std_data = sc.transform(input_data_reshaped)
    std_data = sc.transform(input_data_reshaped)

    print(std_data)

    prediction = modeld.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        print('The person is not diabetic')
    else:
        print('The person is diabetic')
#     input_data = (prg, glc, bp, skt, ins, bmi, dpf, age)

# # changing the input_data to numpy array
#     input_data_as_numpy_array = np.asarray(input_data)

# # reshape the array as we are predicting for one instance
#     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# # standardize the input data
#     std_data = sc.fit_transform(input_data_reshaped)
#     std_data = sc.transform(input_data_reshaped)
#     # print(std_data)
#     prediction = modeld.predict(std_data)
#     print(prediction)

#     #for probability
#     probability = modeld.predict_proba(input_data_reshaped)[:, 1]
#     print(probability)
    

    if (prediction[0] == 0):
        return render_template('diabetes.html', pred='You have Diabetes. Please Consult a Doctor')
    else:
        probability = (float(probability) * 100) 
        rounded = format(probability,".2f")
        return render_template('diabetes.html',
                               pred=f'You do not have Diabetes. Probability of having Diabetes is {rounded}')


@app.route('/predicth', methods=['POST'])
def predicth():
    print([x for x in request.form.values()])
    int_features = [float(x.replace("on", "1").replace("unchecked", "0")) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_prediction = modelh.predict(final_features)
    prediction = modelh.predict_proba(final_features)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    if final_prediction == [1]:
        return render_template('heart.html', pred='You have Heart Disease. Please consult a doctor')
    elif final_prediction == [0]:
        output = str(float(output) * 100) + '%'
        return render_template('heart.html', pred=f'You do not have Heart Disease. '
                                                  f'Probability of having Heart Disease is {output}')

@app.route('/predicthnew', methods=['POST'])
def predicthnew():
    # 'age', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'

    # print(request.form)
    age = float(request.form['age'])
    gender = int(request.form['gender'])
    height = float(request.form['height'])#chest pain type
    weight = float(request.form['weight'])#resting blood pressure
    
    # bmi =  bmi = weight / (height * height)

    ap_hi = int(request.form['ap_hi'])#fasting blood sugar level
    ap_lo = int(request.form['ap_lo'])#resting ecg
    cholesterol = int(request.form['cholesterol'])#5
    gluc = int(request.form['gluc'])#maximun heart rate achieved
    smoke = int(request.form['smoke'])#exercise induced angina
    alco = float(request.form['alco'])#ST depression
    active = int(request.form['active'])#slope of ST 
    vessels = int(request.form['vessels'])#Number of major vessels
    thal = int(request.form['thal'])#thal value

    input_data = (age,gender,height,weight,cholesterol,ap_hi,ap_lo,gluc,smoke,alco,active,vessels,thal)
    
    # Changing data to numpy array
    input_data_arr = np.asarray(input_data)

    # Reshaping the numpy array for only one instance
    input_data_reshaped = input_data_arr.reshape(1,-1)
    prediction = modelh.predict(input_data_reshaped)
    print(prediction)

    # print([x for x in request.form.values()])
    # int_features = [float(x.replace("on", "1").replace("unchecked", "0")) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # final_prediction = modelh.predict(final_features)
    # prediction = modelh.predict_proba(final_features)
    # output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if prediction[0] == 1:
        return render_template('heart.html', pred='You have Heart Disease. Please consult a doctor')
    else:
        #output = str(float(output) * 100) + '%'
        return render_template('heart.html', pred=f'You do not have Heart Disease.')


if __name__ == "__main__":
    app.run(debug=True)
