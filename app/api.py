from fastapi import FastAPI
import pandas as pd
from houses.main import preprocess_new_data, load_model
from houses.params import *

app = FastAPI()
app.state.model = load_model()

@app.get('/')
def root():
    return {"You're doing well": "👍🏻"}

@app.get('/predict')
def predict(
    lot_area,
    built_area,
    bedrooms,
    overall_condition,
    pool,
    garage,
    basement,
    air_conditioning,
    fireplace,
    neighbourhood
    ):

    df = pd.DataFrame({'LotArea': int(lot_area), 'OverallCond': int(overall_condition),
                       'GrLivArea': int(built_area), 'BedroomAbvGr': int(bedrooms),
                       'PoolArea': int(pool), 'GarageArea': int(garage),
                       'BsmtFinSF1': int(basement), 'Fireplaces': int(fireplace),
                       'Neighborhood': neighbourhood, 'CentralAir': int(air_conditioning)},
                      index=[0])

    df_processed = preprocess_new_data(df)
    prediction = app.state.model.predict(df_processed)

    return {"house_price": round(float(prediction), 2)}
