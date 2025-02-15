import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the models
lstm_model = load_model('lstm_model_new.h5')
gru_model = load_model('gru_model_new.h5')

# Load dataset
url = "https://raw.githubusercontent.com/dheeraj5988/stock_market_gen_ai/main/combined_stock_data.csv"
df = pd.read_csv(url)

# Ensure Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Ask user to input the number of days they want to predict
days_to_predict = int(input("Enter the number of days to predict future stock prices: "))

# Filter data for the selected stock symbol
symbol = input("Enter the stock symbol for prediction: ")
filtered_df = df[df['Symbol'] == symbol].copy()
filtered_df.sort_values(by='Date', inplace=True)

# Ensure there is enough data for the selected stock
if filtered_df.empty:
    print(f"No data available for the selected stock: {symbol}")
    exit()

# Prepare the data for prediction
def prepare_data(df, scaler):
    data = df[['Close']].values
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Function to make future predictions using a model
def predict_future_prices(model, data, time_steps, days_to_predict):
    future_predictions = []
    last_data = data[-time_steps:]

    for _ in range(days_to_predict):
        last_data_reshaped = last_data.reshape((1, time_steps, 1))
        predicted_price = model.predict(last_data_reshaped)
        future_predictions.append(predicted_price[0, 0])
        last_data = np.append(last_data[1:], predicted_price, axis=0)

    return np.array(future_predictions)

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare the data
time_steps = 10  # Using 10 previous time steps to predict the next one
scaled_data = prepare_data(filtered_df, scaler)

# Make predictions for the future using LSTM and GRU models
predicted_prices_lstm = predict_future_prices(lstm_model, scaled_data, time_steps, days_to_predict)
predicted_prices_gru = predict_future_prices(gru_model, scaled_data, time_steps, days_to_predict)

# Inverse transform predictions to get actual price values
predicted_prices_lstm_actual = scaler.inverse_transform(predicted_prices_lstm.reshape(-1, 1))
predicted_prices_gru_actual = scaler.inverse_transform(predicted_prices_gru.reshape(-1, 1))

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices_lstm_actual, label='LSTM Predicted Prices', color='red')
plt.plot(predicted_prices_gru_actual, label='GRU Predicted Prices', color='blue')
plt.title(f'{symbol} Stock Price Prediction for Next {days_to_predict} Days')
plt.xlabel('Days')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()

# Display the prediction result to the user
print(f"LSTM Predicted Prices for the next {days_to_predict} days:\n", predicted_prices_lstm_actual)
print(f"GRU Predicted Prices for the next {days_to_predict} days:\n", predicted_prices_gru_actual)