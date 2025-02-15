import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Load your stock data (replace 'your_stock_data.csv' with your dataset)
data = pd.read_csv('/Users/dheerajsmac/Documents/VS_code/Python/gen ai/combined_stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Preprocessing
data = data[['Close']]  # Use 'Close' prices for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training and test datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Generate sequences for CNN input
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # 60 days of data
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, height, width, channels] for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)  # 1 channel for grayscale
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Define the CNN model
def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, kernel_size=(3, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # Output layer
    return model

# Compile the model
cnn_model = build_cnn_model()
cnn_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Train the model
history = cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Plot training & validation loss
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generate predictions
predictions = cnn_model.predict(X_test)

# Inverse transform to get actual price values
predictions = scaler.inverse_transform(predictions)
real_data = scaler.inverse_transform(y_test.reshape(-1, 1))

# Save the trained model
cnn_model.save('cnn_stock_price_model.h5')

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(real_data, color='blue', label='Real Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Real vs Predicted Stock Prices')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
