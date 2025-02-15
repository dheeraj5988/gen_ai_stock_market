from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load models
cnn_model = load_model('models/cnn_stock_price_model.h5')
lstm_model = load_model('models/lstm_model_new.h5')
gru_model = load_model('models/gru_model_new.h5')

# Preprocessing function (adjust as per your preprocessing in models)
def preprocess_data(data):
    # Add your preprocessing logic
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    selected_symbol = request.form['symbol']
    # Load and preprocess data based on selected symbol
    # Here, you'd have your logic for predicting prices
    # For example:
    data = pd.read_csv('combined_stock_data.csv')
    filtered_data = data[data['Symbol'] == selected_symbol]
    # Assume some preprocessing function
    processed_data = preprocess_data(filtered_data)
    
    # Make predictions (adjust based on the model and input)
    predictions = cnn_model.predict(processed_data)  # Replace with model you want to use
    predictions = predictions.flatten()  # Reshape if needed

    # Render results
    return render_template('result.html', predictions=predictions, symbol=selected_symbol)

if __name__ == '__main__':
    app.run(debug=True)
