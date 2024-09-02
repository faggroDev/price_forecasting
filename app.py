from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

app = FastAPI()

# Load the dataset and scalers
df = pd.read_csv('data/preprocessed_cleand_price_data_tn.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df.sort_values(by=['District Name', 'Market Name', 'Commodity', 'Variety', 'date'])

# Load scalers
scalers = {}
for price_type in ['Min Price', 'Max Price', 'Modal Price']:
    scalers[price_type] = joblib.load(f'data/scaler_api{price_type}.pkl')

# Define the create_dataset function
def create_dataset(data, look_back=30):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

# Filter dataset based on the look_back period
Market_data = {}
look_back = 30
valid_rows = []

for district in df['District Name'].unique():
    district_df = df[df['District Name'] == district]
    for market in district_df['Market Name'].unique():
        market_df = district_df[district_df['Market Name'] == market]
        for commodity in market_df['Commodity'].unique():
            commodity_df = market_df[market_df['Commodity'] == commodity]
            for variety in commodity_df['Variety'].unique():
                variety_df = commodity_df[commodity_df['Variety'] == variety]
                if len(variety_df) > look_back:
                    X, Y = {}, {}
                    for price_type in ['Min Price', 'Max Price', 'Modal Price']:
                        prices = variety_df[f'{price_type}_scaled'].values
                        X[price_type], Y[price_type] = create_dataset(prices, look_back)
                        X[price_type] = X[price_type].reshape(X[price_type].shape[0], X[price_type].shape[1], 1)
                    Market_data[(district, market, commodity, variety)] = {'X': X, 'Y': Y, 'dates': variety_df['date'].values}
                    valid_rows.extend(variety_df.index.tolist())
                else:
                    print(f"Insufficient data for Market: {market}, District: {district}, Commodity: {commodity}, Variety: {variety}.")

df_cleaned = df.loc[valid_rows].copy()

def build_and_train_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
    return model

def forecast_prices(model, market_key, start_date, num_days, price_type):
    scaler = scalers[price_type]
    last_sequence = Market_data[market_key]['X'][price_type][-1].copy()
    forecasts = []
    forecast_dates = [start_date + timedelta(days=i) for i in range(num_days)]
    for i in range(num_days):
        prediction = model.predict(last_sequence.reshape(1, look_back, 1))
        forecasts.append(prediction[0][0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    return forecast_dates, forecasts
@app.get("/")
def read_root():
    return {"message": "Welcome to the Grain Price Forecasting API. Use the appropriate endpoints to get data."}


@app.get("/forecast/")
def get_forecast(
    district: str, 
    market: str, 
    commodity: str, 
    variety: str, 
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"), 
    num_days: int = Query(30, description="Number of days to forecast")
):
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    market_key = (district, market, commodity, variety)
    if market_key not in Market_data:
        raise HTTPException(status_code=404, detail="Data not available for the given market, district, commodity, and variety combination")

    forecasts_dict = {}
    forecast_dates = {}

    for price_type in ['Min Price', 'Max Price', 'Modal Price']:
        X_train, X_test, Y_train, Y_test = train_test_split(
            Market_data[market_key]['X'][price_type],
            Market_data[market_key]['Y'][price_type],
            test_size=0.2,
            random_state=42
        )
        model = build_and_train_model(X_train, Y_train, X_test, Y_test)
        forecast_dates[price_type], forecasts_dict[price_type] = forecast_prices(model, market_key, start_date, num_days, price_type)

    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates['Modal Price']],
        "min_price_forecast": forecasts_dict['Min Price'].flatten().tolist(),
        "max_price_forecast": forecasts_dict['Max Price'].flatten().tolist(),
        "modal_price_forecast": forecasts_dict['Modal Price'].flatten().tolist(),
    }
