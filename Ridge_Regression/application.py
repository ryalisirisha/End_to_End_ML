from flask import Flask,request,render_template,jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge  

application=Flask(__name__)
app = application 

# import the Ridge and scaler pickle files
ridge_model =pickle.load(open('models/model.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        
        ISI=float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data=scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_scaled_data)
        return render_template('home.html',results = result[0])

                   
    else:
        return render_template('home.html')
    return 
  


if __name__=="__main__":
    app.run(host='0.0.0.0')
