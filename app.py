from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST']) # type: ignore
def predict_datapoint():
    if request.method == 'POST':
        facing = request.form.get('facing')
        rate = float(request.form.get('rate')) # type: ignore
        area_sqft = float(request.form.get('area_sqft')) # type: ignore
        bedRoom = int(request.form.get('bedRoom')) # type: ignore
        bathroom = int(request.form.get('bathroom')) # type: ignore
        balcony = int(request.form.get('balcony')) # type: ignore
        noOfFloor = int(request.form.get('noOfFloor')) # type: ignore
        agePossession_processed = float(request.form.get('agePossession_processed')) # type: ignore
        avg_rating = float(request.form.get('avg_rating')) # type: ignore

        custom_data = CustomData(facing, rate, area_sqft, bedRoom, bathroom, balcony, noOfFloor, agePossession_processed, avg_rating) # type: ignore
        data_frame = custom_data.get_data_as_dataframe()

        prediction_pipeline = PredictionPipeline()
        result = prediction_pipeline.predict(data_frame)

        return render_template('home.html', prediction_text=f'Predicted Price: {result[0]}')