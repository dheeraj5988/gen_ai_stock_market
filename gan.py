import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load your dataset (replace with actual dataset path)
data = pd.read_csv('combined_stock_data.csv')  # Ensure to specify the correct file extension
print(f"Loaded dataset with shape: {data.shape}")

# ----------------------
# DATA PREPROCESSING
# ----------------------
def preprocess_data(data):
    data = data[['Close']].values  # Use only 'Close' prices
    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize to [0, 1]
    return data

stock_data = preprocess_data(data)

# Hyperparameters
latent_dim = 100
epochs = 1000
batch_size = 64

# ----------------------
# BUILDING THE GENERATOR
# ----------------------
def build_generator():
    model = Sequential()

    # Input Layer
    model.add(Input(shape=(latent_dim,)))  # latent_dim = 100

    # Hidden Layers with LeakyReLU activations
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))

    # Output Layer
    model.add(Dense(stock_data.shape[1], activation='sigmoid'))  # Use sigmoid activation
    model.add(Reshape((stock_data.shape[1],)))  # Ensure correct output shape

    return model

generator = build_generator()
print("Generator Summary:")
generator.summary()

# ------------------------
# BUILDING THE DISCRIMINATOR
# ------------------------
def build_discriminator():
    model = Sequential()

    # Input Layer with Flattening
    model.add(Input(shape=(stock_data.shape[1],)))  # Match shape with generator output
    model.add(Flatten())

    # Hidden Layers with LeakyReLU activations
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))

    # Output Layer for Binary Classification
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
print("Discriminator Summary:")
discriminator.summary()

# ----------------------
# BUILDING THE GAN MODEL
# ----------------------
discriminator.trainable = False  # Freeze discriminator during generator training

gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# ----------------------
# HELPER FUNCTION TO FETCH REAL DATA
# ----------------------
def get_real_data(batch_size):
    idx = np.random.randint(0, stock_data.shape[0], batch_size)
    return stock_data[idx]

# ----------------------
# TRAINING THE GAN MODEL
# ----------------------
def train_gan(epochs, batch_size):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train Discriminator on Real Data
        real_data = get_real_data(half_batch)
        real_labels = np.ones((half_batch, 1))  # Real data labeled as 1
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)

        # Train Discriminator on Fake Data
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_data = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))  # Fake data labeled as 0
        d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)

        # Average the Discriminator Loss
        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])  # Ensure this is a float

        # Calculate average accuracy
        d_accuracy = 0.5 * ((d_loss_real[1] if len(d_loss_real) > 1 else 0) + 
                            (d_loss_fake[1] if len(d_loss_fake) > 1 else 0))

        # Train Generator to Fool the Discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))  # Trick discriminator with real labels
        g_loss = gan.train_on_batch(noise, valid_labels)  # g_loss is a float

        # Print Progress Every 100 Epochs
        if epoch % 100 == 0:
                print(f"Type of d_loss: {type(d_loss)}, Type of d_accuracy: {type(d_accuracy)}, Type of g_loss: {type(g_loss)}")
    print(f"{epoch}/{epochs} [D loss: {d_loss:.4f}, D acc.: {d_accuracy * 100:.2f}%] [G loss: {g_loss:.4f}]")


# Start Training
print("Starting GAN Training...")
train_gan(epochs, batch_size)



def generate_data(num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise)
    return generated_data

def plot_data(real_data, generated_data):
    plt.figure(figsize=(12, 6))
    plt.plot(real_data, label='Real Data', alpha=0.5)
    plt.plot(generated_data, label='Generated Data', alpha=0.5)
    plt.title('Real vs Generated Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()

# Start Training
print("Starting GAN Training...")
generated_data = train_gan(epochs, batch_size)

# Plot the real and generated data
plot_data(stock_data, generated_data)