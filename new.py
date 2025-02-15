#code with the accuracy of the models

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam

# Load dataset
url1 = "https://raw.githubusercontent.com/dheeraj5988/stock_market_gen_ai/main/combined_stock_data.csv"
df = pd.read_csv(url1)

# Ensure Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Get unique stock symbols
symbols = df['Symbol'].unique()

# Display stock symbols for user to select
print("Select any one option from the given stocks below:\n")
for idx, symbol in enumerate(symbols):
    print(f"{idx + 1}. {symbol}")

# Ask user to select a stock
try:
    user_choice = int(input("\nEnter the number corresponding to the stock symbol you want to predict: ")) - 1
    if user_choice < 0 or user_choice >= len(symbols):
        raise ValueError("Invalid number")
    selected_symbol = symbols[user_choice]
    print(f"\nYou have selected: {selected_symbol}")
except ValueError:
    print("\nInvalid input. Please enter a valid number.")
    exit()

# Filter the DataFrame for the selected stock
filtered_df = df[df['Symbol'] == selected_symbol].copy()
filtered_df.sort_values(by='Date', inplace=True)

if filtered_df.empty:
    print(f"No data available for the selected stock: {selected_symbol}.")
    exit()

# Plot the stock prices with grid
plt.figure(figsize=(10, 4))
plt.plot(filtered_df['Date'], filtered_df['Close'], label=f'{selected_symbol} stock', color='blue')
plt.xlabel("Date")
plt.ylabel("INR")
plt.title(f"{selected_symbol} Stock Price")
date_form = DateFormatter("%Y")
plt.gca().xaxis.set_major_formatter(date_form)
plt.legend()
plt.grid(True)
plt.show()

# Calculate technical indicators
def get_technical_indicators(data):
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    return data.dropna()

# Apply technical indicators
T_df = get_technical_indicators(filtered_df)
data = T_df[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Prepare data for LSTM/GRU
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = prepare_data(data, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

def evaluate_model(model, X, y, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(actual, predictions)
    accuracy = 100 - mse  # Simplified accuracy metric
    return accuracy, predictions

# LSTM Model Handling
lstm_model_path = 'lstm_model_new.h5'
if os.path.exists(lstm_model_path):
    print("LSTM model found. Loading the model...")
    lstm_model = load_model(lstm_model_path)
else:
    print("No LSTM model found. Training a new model...")
    lstm_model = Sequential([
        Input(shape=(time_steps, 1)),
        LSTM(70, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(70, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=1200, batch_size=50)
    lstm_model.save(lstm_model_path)

lstm_accuracy, predicted_prices_lstm = evaluate_model(lstm_model, X, y, scaler)
print(f"LSTM Model Accuracy: {lstm_accuracy:.2f}%")

# Plot LSTM predictions vs actual prices with grid
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(y.reshape(-1, 1)), label='Actual Prices', color='green')
plt.plot(predicted_prices_lstm, label='LSTM Predicted Prices', color='red')
plt.title(f'{selected_symbol} Stock Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# GRU Model Handling
gru_model_path = 'gru_model_new.h5'
if os.path.exists(gru_model_path):
    print("GRU model found. Loading the model...")
    gru_model = load_model(gru_model_path)
else:
    print("No GRU model found. Training a new model...")
    gru_model = Sequential([
        Input(shape=(time_steps, 1)),
        GRU(70, activation='relu', return_sequences=True),
        Dropout(0.2),
        GRU(70, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X, y, epochs=300, batch_size=42)
    gru_model.save(gru_model_path)

gru_accuracy, predicted_prices_gru = evaluate_model(gru_model, X, y, scaler)
print(f"GRU Model Accuracy: {gru_accuracy:.2f}%")

# Plot GRU predictions vs actual prices with grid
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(y.reshape(-1, 1)), label='Actual Prices', color='green')
plt.plot(predicted_prices_gru, label='GRU Predicted Prices', color='blue')
plt.title(f'{selected_symbol} Stock Price Prediction with GRU')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()



from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Function to calculate RMSE and MAE
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

# Predict with the LSTM model
print("LSTM model found. Loading the model...")
lstm_model = load_model(lstm_model_path)
predicted_prices_lstm = lstm_model.predict(X)
predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Evaluate the LSTM model
print("Evaluating LSTM Model...")
evaluate_model(actual_prices, predicted_prices_lstm)

# Predict with the GRU model
print("GRU model found. Loading the model...")
gru_model = load_model(gru_model_path)
predicted_prices_gru = gru_model.predict(X)
predicted_prices_gru = scaler.inverse_transform(predicted_prices_gru)

# Evaluate the GRU model
print("Evaluating GRU Model...")
evaluate_model(actual_prices, predicted_prices_gru)

 # Plotting the results for LSTM
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices', color='green')
plt.plot(predicted_prices_lstm, label='LSTM Predicted Prices', color='red')
plt.title(f'{selected_symbol} Stock Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)  # Add grid to the plot
plt.legend()
plt.show()

# # Plotting the results for GRU
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices', color='green')
plt.plot(predicted_prices_gru, label='GRU Predicted Prices', color='blue')
plt.title(f'{selected_symbol} Stock Price Prediction with GRU')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)  # Add grid to the plot
plt.legend()
plt.show()




