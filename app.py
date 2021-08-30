from  flask import Flask, render_template, request
import pickle
import pandas as pd 
import numpy as np 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
gradientboost = pickle.load(open('GB.pkl', 'rb'))
df_old= pd.read_csv("processed_data")



@app.route('/')
@app.route('/home', methods=['GET'])
def home_page():
    return render_template('home.html')
def get_data():
    Account_Length = request.form.get('Account_Length')
    Vmail_Message= request.form.get('Vmail_Message')
    Day_Mins= request.form.get('Day_Mins')   
    Eve_Mins= request.form.get('Eve_Mins')  
    Night_Mins = request.form.get('Night_Mins') 
    Intl_Mins = request.form.get('Intl_Mins') 
    CustServ_Calls = request.form.get('CustServ_Calls')
    Intl_Plan  = request.form.get('Intl_Plan')
    Vmail_Plan = request.form.get('Vmail_Plan')
    Day_Calls = request.form.get('Day_Calls') 
    Day_Charge=request.form.get('Day_Charge') 
    Eve_Calls = request.form.get('Eve_Calls')
    Eve_Charge = request.form.get('Eve_Charge')
    Night_Calls= request.form.get('Night_Calls')
    Night_Charge =request.form.get('Night_Charge')
    Intl_Calls =request.form.get('Intl_Calls')
    Intl_Charge = request.form.get('Intl_Charge')
    data =[[Account_Length,Vmail_Message,Day_Mins,Eve_Mins,Night_Mins,Intl_Mins,CustServ_Calls,
            Intl_Plan,Vmail_Plan,Day_Calls,Day_Charge,Eve_Calls,Eve_Charge,Night_Calls,Night_Charge,
            Intl_Calls,Intl_Charge]]
    new_df = pd.DataFrame(data, columns = ['Account_Length','Vmail_Message','Day_Mins','Eve_Mins','Night_Mins','Intl_Mins','CustServ_Calls',
            'Intl_Plan','Vmail_Plan','Day_Calls','Day_Charge','Eve_Calls','Eve_Charge','Night_Calls','Night_Charge',
            'Intl_Calls','Intl_Charge'])
    for col in new_df.columns:
        print(new_df[col])
    #Scaling

    df = pd.concat([df_old,new_df], ignore_index=True)
    col_scale = ['Account_Length','Day_Mins','Eve_Mins','Night_Mins','Intl_Mins','CustServ_Calls','Day_Calls','Day_Charge','Eve_Calls','Eve_Charge',
 'Night_Calls','Night_Charge','Intl_Calls','Intl_Charge']
    scaler = StandardScaler()
    df[col_scale]= scaler.fit_transform(df[col_scale])  
    for col in df.columns :
        print(df[col].tail(1))
    return df
@app.route('/send', methods=['POST'])
def show_data():
    df = get_data().tail(1)
    prediction = gradientboost.predict(df)
    outcome = 'Churner'
    if prediction == 0:
        outcome = 'Non-Churner'

    return render_template('results.html',
                           result = outcome)