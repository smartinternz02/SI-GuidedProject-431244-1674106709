from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open(r'C:\Users\dell\Telegram Desktop\Smart Home - Temperature Prediction\Smart Home - Temperature Prediction\Flask\temperature.pkl','rb'))
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
    input_feature= [float(x) for x in request.form.values()] 
    input_feature=[np.array(input_feature)]
    print(input_feature)
    names = ['CO2_room', 'Relative_humidity_room', 'Lighting_room', 'Meteo_Rain', 'Meteo_Wind', 'Meteo_Sun_light_in_west_facade',
       'Outdoor_relative_humidity_Sensor']
    print(names)
    data = pd.DataFrame(input_feature,columns=names)
    print(data)
    prediction=model.predict(data)
    print(prediction)
    return render_template('predict.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug = True)