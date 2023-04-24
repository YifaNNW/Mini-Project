import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Input
from keras.layers import Dense, TimeDistributed
from keras.utils.np_utils import *
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('2024-01-02LOBs.csv')
# move 'mid_price' to the last column
mid_price = data['mid_price']
data.drop(labels=['mid_price'], axis=1, inplace=True)
data.insert(len(data.columns), 'mid_price', mid_price)

data['next_mid_price'] = data['mid_price'].shift(-1)
data.dropna(inplace=True)
data['price_movement'] = np.where(data['next_mid_price'] > data['mid_price'], 1, 0)
data.drop(['next_mid_price'], axis=1, inplace=True)

# print(data.shape)
# print(data.columns)
data = data.values
y = np.expand_dims(data[:, -1], axis=1)
X = data[:, :-1]

y_binary = to_categorical(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

batch_size = 128


def reshape_to_batches(data, batch_size):
    # Pad the data with zeros to make it fit the batch_size
    if data.shape[0] % batch_size != 0:
        data = np.concatenate((data, np.zeros((batch_size - data.shape[0] % batch_size, data.shape[1]))))
    # Reshape the data to batches
    data = data.reshape((int(data.shape[0] / batch_size), batch_size, data.shape[1]))
    return data


X_train_batch = reshape_to_batches(X_train, batch_size)
y_train_batch = reshape_to_batches(y_train, batch_size)
X_test_batch = reshape_to_batches(X_test, batch_size)
y_test_batch = reshape_to_batches(y_test, batch_size)

# Create the model
inputs = Input(shape=(batch_size, X_train.shape[1]))
lstm = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)

predictions = TimeDistributed(Dense(2, activation='softmax'))(lstm)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_batch, y_train_batch, epochs=10, batch_size=batch_size)

# Evaluate the model
y_pred = model.predict(X_test_batch)
print(y_pred)
print(y_train_batch)

y_pred[np.where(y_pred >= 0.5)] = 1
y_pred[np.where(y_pred < 0.5)] = 0
print(classification_report(
    y_test_batch.reshape(y_test_batch.shape[0] * y_test_batch.shape[1], y_test_batch.shape[2])[:, 1],
    y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])[:, 1],
    target_names=["Down", "Up"],
    digits=5))

# Save the model
model.save('LSTM.h5')



