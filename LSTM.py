import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Input
from keras.layers import Dense, TimeDistributed
from keras.utils.np_utils import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('2024-01-02LOBs.csv')
# move 'mid_price' to the last column
mid_price = data['mid_price']
data.drop(labels=['mid_price'], axis=1, inplace=True)
data.insert(len(data.columns), 'mid_price', mid_price)
data['next_mid_price'] = data['mid_price'].shift(-1)
data.dropna(inplace=True)
data['price_movement'] = np.where(data['next_mid_price'] > data['mid_price'], 1, 0)
data.drop(['next_mid_price'], axis=1, inplace=True)

data = data.values
y = np.expand_dims(data[:, -1], axis=1)
X = data[:, :-1]


# split data into train and test
def train_test_split(data, test_size=0.2):
    n = data.shape[0]
    idx_split = int(n*(1-test_size))
    train = data[0:idx_split, :]
    test = data[idx_split:n, :]
    return train, test


y_binary = to_categorical(y)

pipeline = Pipeline([('scaler', StandardScaler())])
X = pipeline.fit_transform(X)

X_train, X_test = train_test_split(X, test_size=0.2)
y_train, y_test = train_test_split(y_binary, test_size=0.2)

batch_size = 128


def reshape_to_batches(a, batch_size):
    #pad if the length is not divisible by the batch_size
    batch_num = np.ceil(a.shape[0] / float(batch_size))
    modulo = batch_num * batch_size - a.shape[0]
    if modulo != 0:
        pad = np.zeros((int(modulo), a.shape[1]))
        a = np.vstack((a, pad))
    return np.array(np.split(a, batch_num))


X_train_batch = reshape_to_batches(X_train, batch_size)
y_train_batch = reshape_to_batches(y_train, batch_size)
X_test_batch = reshape_to_batches(X_test, batch_size)
y_test_batch = reshape_to_batches(y_test, batch_size)


def _3d_to_2d(arr):
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])


def classification_result(y_pred, thresh=0.5):
  cutt_off_tr = thresh # some threshold

  y_pred[np.where(y_pred >= cutt_off_tr)] = 1
  y_pred[np.where(y_pred < cutt_off_tr)]  = 0

  print(confusion_matrix(
         _3d_to_2d(y_test_batch)[:, 1],
         _3d_to_2d(y_pred)[:, 1]))

  print()
  print(classification_report(
          _3d_to_2d(y_test_batch)[:, 1],
          _3d_to_2d(y_pred)[:, 1],
          target_names = ["Down", "Up"],
          digits = 5))


# Create the model
inputs = Input(shape=(batch_size, X_train.shape[1]))
lstm = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)

predictions = TimeDistributed(Dense(2, activation='softmax'))(lstm)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_batch, y_train_batch, epochs=10, batch_size=batch_size)

y_pred = model.predict(X_test_batch)
classification_result(y_pred)


