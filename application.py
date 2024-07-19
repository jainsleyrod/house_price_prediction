from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            area = int(request.form['area']),
            typeOfSale = request.form['typeOfSale'],
            region = request.form['region'],
            floorRange = request.form['floorRange'],
            Year = int(request.form['Year']),
            mktSegment = request.form['mktSegment'],
            propertyType = request.form['propertyType'],
            typeOfArea=request.form['typeOfArea']
        )

        df = data.get_data_as_df()
        print(df)

        pipeline = PredictPipeline()
        results = pipeline.predict(df)

        return render_template('home.html',results=results[0])
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
