import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
data= pd.read_csv('train.csv')
model = pickle.load(open("MachineKnight.pkl","rb"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])

def predict():
    locality = sorted(data['locality'].unique())
    lease_type = sorted(data['lease_type'].unique())
    water_supply = sorted(data['water_supply'].unique())
    negotiable= sorted(data['negotiable'].unique())
    lift= sorted(data['lift'].unique())
    gym= sorted(data['gym'].unique())
    parking= sorted(data['parking'].unique())
    swimming_pool= sorted(data['swimming_pool'].unique())
    furnishing= sorted(data['furnishing'].unique())

    return render_template('predict.html',locality = locality,lease_type=lease_type,water_supply=water_supply,negotiable=negotiable,
                           lift=lift,gym=gym,parking=parking,swimming_pool=swimming_pool,furnishing=furnishing)

@app.route('/index',methods=['POST','GET'])
def index():
    return render_template('index.html')


@app.route('/result',methods=['POST'])
def result():
    locality=request.form.get('locality')
    lease_type=request.form.get('lease_type')
    water_supply=request.form.get('water_supply')
    sqft=request.form.get('sqft')
    negotiable = request.form.get('negotiable')
    bathroom = request.form.get('bathroom')
    balconies = request.form.get('balconies')
    lift = request.form.get('lift')
    gym = request.form.get('gym')
    parking = request.form.get('bathrooms')
    swimming_pool = request.form.get('swimming_pool')
    furnishing = request.form.get('furnishing')



    prediction= model.predict(pd.DataFrame(columns=['lease_type', 'gym', 'lift', 'swimming_pool', 'negotiable','furnishing','parking','property_size','bathroom','water_supply','balconies'],
                              data=np.array([lease_type,gym,lift,swimming_pool,negotiable,furnishing,parking,sqft,bathroom,water_supply,balconies]).reshape(1, 11)))
    print(prediction)

    return str(np.round(prediction[0],2))


if __name__=="__main__":
    app.run(debug=True, port=5001)
