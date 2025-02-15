import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm

sns.set()
tf.compat.v1.random.set_random_seed(1234)

# Load your dataset
df = pd.read_csv('/Users/dheerajsmac/Documents/VS_code/Python/gen ai/combined_stock_data.csv')  # Ensure the path is correct
df.head()

# Assuming the dataset has a 'Close' column
minmax = MinMaxScaler().fit(df[['Close']].astype('float32'))
df_log = minmax.transform(df[['Close']].astype('float32'))
df_log = pd.DataFrame(df_log)

# Split train and test datasets
test_size = 30
simulation_size = 10

df_train = df_log.iloc[:-test_size]
df_test = df_log.iloc[-test_size:]

# Define the LSTM Model
class Model:
    def __init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias=0.1):
        def lstm_cell(size_layer):
            return tf.keras.layers.GRU(size_layer)

        rnn_cells = [lstm_cell(size_layer) for _ in range(num_layers)]
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        self.hidden_layer = tf.placeholder(tf.float32, (None, num_layers * size_layer))
        
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            tf.keras.layers.StackedRNNCells(rnn_cells), self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

# LSTM training and forecasting function
def forecast():
    tf.reset_default_graph()
    modelnn = Model(learning_rate=0.01, num_layers=1, size=df_log.shape[1], size_layer=128, output_size=df_log.shape[1])
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    pbar = tqdm(range(300), desc='train loop')
    for i in pbar:
        init_value = np.zeros((1, 128))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, 5):
            index = min(k + 5, df_train.shape[0] - 1)
            batch_x = np.expand_dims(df_train.iloc[k:index, :].values, axis=0)
            batch_y = df_train.iloc[k + 1:index + 1, :].values
            logits, last_state, _, loss = sess.run(
                [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                feed_dict={
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))

    output_predict = np.zeros((df_train.shape[0] + test_size, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    
    init_value = np.zeros((1, 128))

    for k in range(0, (df_train.shape[0] // 5) * 5, 5):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(df_train.iloc[k:k + 5], axis=0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1:k + 6] = out_logits

    future_day = test_size - 1
    for i in range(future_day):
        o = output_predict[-future_day - 5:-future_day]
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(o, axis=0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]

    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.3)

    return deep_future[-test_size:]

results = []
for i in range(simulation_size):
    print('simulation %d' % (i + 1))
    results.append(forecast())

# Plotting the results
accuracies = [calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]
plt.figure(figsize=(15, 5))
for no, r in enumerate(results):
    plt.plot(r, label='forecast %d' % (no + 1))
plt.plot(df['Close'].iloc[-test_size:].values, label='true trend', c='black')
plt.legend()
plt.title('Average accuracy: %.4f' % (np.mean(accuracies)))
plt.show()
