import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, GRU, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam  # Import Adam optimizer

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

# Plot the stock prices
plt.figure(figsize=(10, 4))
plt.plot(filtered_df['Date'], filtered_df['Close'], label=f'{selected_symbol} stock', color='blue')
plt.xlabel("Date")
plt.ylabel("INR")
plt.title(f"{selected_symbol} Stock Price")
date_form = DateFormatter("%Y")
plt.gca().xaxis.set_major_formatter(date_form)
plt.legend()
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

# Reshape X for LSTM/GRU input
X = X.reshape((X.shape[0], X.shape[1], 1))

# Early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Check if the LSTM model exists, load it if available, otherwise train
lstm_model_path = 'lstm_model_new.h5'
if os.path.exists(lstm_model_path):
    print("LSTM model found. Loading the model...")
    lstm_model = load_model(lstm_model_path)
else:
    print("No LSTM model found. Training a new model...")
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(time_steps, 1)))
    lstm_model.add(LSTM(70, activation='relu', return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(70, activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stopping])
    lstm_model.save(lstm_model_path)

# Predict with the LSTM model
predicted_prices_lstm = lstm_model.predict(X)
predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Plot LSTM predictions vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices', color='green')
plt.plot(predicted_prices_lstm, label='LSTM Predicted Prices', color='red')
plt.title(f'{selected_symbol} Stock Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Check if the GRU model exists, load it if available, otherwise train
gru_model_path = 'gru_model_new.h5'
if os.path.exists(gru_model_path):
    print("GRU model found. Loading the model...")
    gru_model = load_model(gru_model_path)
else:
    print("No GRU model found. Training a new model...")
    gru_model = Sequential()
    gru_model.add(Input(shape=(time_steps, 1)))
    gru_model.add(GRU(70, activation='relu', return_sequences=True))
    gru_model.add(Dropout(0.2))
    gru_model.add(GRU(70, activation='relu'))
    gru_model.add(Dropout(0.2))
    gru_model.add(Dense(1))

    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stopping])
    gru_model.save(gru_model_path)

# Predict with the GRU model
predicted_prices_gru = gru_model.predict(X)
predicted_prices_gru = scaler.inverse_transform(predicted_prices_gru)

# Plot GRU predictions vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices', color='green')
plt.plot(predicted_prices_gru, label='GRU Predicted Prices', color='blue')
plt.title(f'{selected_symbol} Stock Price Prediction with GRU')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()




# Discriminator Model
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])  # Fixed Adam import
    return model

# Generator Model
def build_generator(latent_dim, output_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='tanh'))  # Tanh for continuous output
    return model

# GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator when training the GAN
    gan_input = Input(shape=(latent_dim,))
    generated_output = generator(gan_input)
    gan_output = discriminator(generated_output)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam())  # Fixed Adam import
    return gan

# Hyperparameters
latent_dim = 100  # Latent space dimension
epochs = 50
batch_size = 128

# Load your dataset (example)
def load_data():
    # Replace this with your actual stock data
    data = np.random.normal(0, 1, (1000, 10))  # Example dataset with 10 features
    return data

# Generate fake stock data
def generate_fake_data(generator, batch_size):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))  # Generate random noise
    return generator.predict(noise)

# Training GAN
def train_gan(epochs, batch_size):
    data = load_data()
    input_shape = data.shape[1]

    # Create models
    discriminator = build_discriminator(input_shape)
    generator = build_generator(latent_dim, input_shape)
    gan = build_gan(generator, discriminator)

    for epoch in range(epochs):
        # Train Discriminator
        real_data = data[np.random.randint(0, data.shape[0], size=batch_size)]
        fake_data = generate_fake_data(generator, batch_size)
        
        X_combined = np.concatenate([real_data, fake_data])
        y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])  # Labels: 1 for real, 0 for fake
        
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y_mislabeled = np.ones((batch_size, 1))  # The generator wants to fool the discriminator

        g_loss = gan.train_on_batch(noise, y_mislabeled)

        # Print the progress
        print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}%] [G loss: {g_loss}]")

    # Save models
    generator.save("generator_model.keras")
    discriminator.save("discriminator_model.keras")

# Train the GAN
train_gan(epochs, batch_size)