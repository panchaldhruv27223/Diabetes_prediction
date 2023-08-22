from flask import Flask, url_for,render_template,request
from urllib.request import urlopen
import requests
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app = application

scaler = pickle.load(open(r"C:\Users\dhruv\Documents\ML\ML_pw\Logistic_project_1\models\scaler.pkl", "rb"))
logistic_model = pickle.load(open(r"C:\Users\dhruv\Documents\ML\ML_pw\Logistic_project_1\models\logisticmodel.pkl","rb"))

@app.route("/")
def my_first():
    return render_template("index.html")


@app.route("/second",methods = ["GET","POSt"])
def my_second():
    return render_template("menu.html")

@app.route("/menu", methods =["GET","POST"])
def my_third():

    if (request.method == "POST"):
        
        Pregnancies = float(request.form.get("Pregnancies"))    
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))

        x_scale = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        y_predict = logistic_model.predict(x_scale)[0]

        if y_predict:
            return render_template("result.html" , result = "You Have Diabeties")
        else :
            return render_template("result.html",result = "You not having a Diabeties")

    else :
        return render_template("menu.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)