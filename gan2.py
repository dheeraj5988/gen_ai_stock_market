import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Load your stock data
data = pd.read_csv('/Users/dheerajsmac/Documents/VS_code/Python/gen ai/combined_stock_data.csv')  # Adjust the path to your data file
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Adjust based on your dataset
data = data.values.astype('float32')

# Normalize the data
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Define parameters
noise_dim = 10  # Dimensionality of the noise input for the generator
batch_size = 32
epochs = 5000
half_batch = batch_size // 2

# Build Generator
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(5, activation='sigmoid'))  # 5 outputs for Open, High, Low, Close, Volume
    return model

# Build Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))  # Output for real/fake
    return model

# Compile models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(noise_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training GAN
for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, data.shape[0], half_batch)
    real_data = data[idx]

    noise = np.random.normal(0, 1, (half_batch, noise_dim))
    fake_data = generator.predict(noise)

    # Labels
    real_labels = np.ones((half_batch, 1))
    fake_labels = np.zeros((half_batch, 1))

    # Train on real and fake data
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    g_loss = gan.train_on_batch(noise, real_labels)  # Try to trick the discriminator

    # Print losses
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss[0]}')

# Save models
generator.save('stock_price_generator.h5')
discriminator.save('stock_price_discriminator.h5')
