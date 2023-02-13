from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os


import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "Sdn3hgmtLyEKEKZaQZi2588O6naLorrry4o3uhrIpFyi"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route('/output', methods = ['post','get'])
def output():
    #  reading the inputs given by the user
    co2 = float(request.form['CO2_room'])
    ihumidity = float(request.form['Relative_humidity_room'])
    light = float(request.form['Lighting_room'])
    rain = float(request.form['Meteo_Rain'])
    wind = float(request.form['Meteo_Wind'])
    sunlight = float(request.form['Meteo_Sun_light_in_west_facade'])
    ohumidity = float(request.form['Outdoor_relative_humidity_Sensor'])


    input_feature = [[co2,ihumidity,light,rain,wind,sunlight,ohumidity]]

    payload_scoring = {"input_data": [{"field":[["CO2_room", "Relative_humidity_room", "Lighting_room", "Meteo_Rain", "Meteo_Wind", "Meteo_Sun_light_in_west_facade",
       "Outdoor_relative_humidity_Sensor"]], "values": input_feature}]}
    #names = ['CO2_room', 'Relative_humidity_room', 'Lighting_room', 'Meteo_Rain', 'Meteo_Wind', 'Meteo_Sun_light_in_west_facade','Outdoor_relative_humidity_Sensor']

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/f3e34985-ab4a-4d54-a11a-e6831979ec74/predictions?version=2022-06-16', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    prediction = response_scoring.json()
    pred = prediction['predictions'][0]['values'][0][0]
    print(pred)
    pred = np.round(pred,2)
    return render_template('predict.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug = True)