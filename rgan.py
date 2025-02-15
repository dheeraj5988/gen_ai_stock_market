import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm

sns.set()
tf.compat.v1.random.set_random_seed(1234)

# Load your dataset
df = pd.read_csv('/Users/dheerajsmac/Documents/VS_code/Python/gen ai/combined_stock_data.csv')
# Assume 'Close' is the column you want to forecast
df['Close'] = df['Close'].astype('float32')

# MinMax Scaling
minmax = MinMaxScaler().fit(df[['Close']])
df_scaled = minmax.transform(df[['Close']])
df_scaled = pd.DataFrame(df_scaled)

# Split train and test
test_size = 30
df_train = df_scaled[:-test_size]
df_test = df_scaled[-test_size:]

class GAN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=self.input_dim))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='tanh'))  # Output layer
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer
        return model

    def build_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.discriminator.trainable = False  # Freeze the discriminator during generator training
        model = tf.keras.Sequential([self.generator, self.discriminator])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, X_train, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            # Train the discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            fake_data = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Initialize and train the GAN
gan = GAN(input_dim=5)  # Example input dimension; adjust based on your needs
gan.train(df_train.values, epochs=1000, batch_size=32)

# Forecasting using the generator
def forecast(gan, num_samples):
    noise = np.random.normal(0, 1, (num_samples, 5))  # Adjust input shape based on generator
    generated_data = gan.generator.predict(noise)
    generated_data = minmax.inverse_transform(generated_data)  # Inverse transform to original scale
    return generated_data

# Generate future stock price predictions
future_predictions = forecast(gan, 30)

# Plot results
plt.figure(figsize=(15, 5))
plt.plot(df['Close'].values[-test_size:], label='True Prices', color='black')
plt.plot(future_predictions, label='Predicted Prices', color='blue')
plt.legend()
plt.title('Stock Price Predictions using GAN')
plt.show()
