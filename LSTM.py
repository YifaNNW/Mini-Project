import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Input
from keras.layers import Dense, TimeDistributed
from keras.utils.np_utils import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# parameters
batch_size = 128
epochs = 50
percentage = 0.7

# data = pd.read_csv('2024-01-02LOBs.csv')
# # move 'mid_price' to the last column
# mid_price = data['mid_price']
# data.drop(labels=['mid_price'], axis=1, inplace=True)
# data.insert(len(data.columns), 'mid_price', mid_price)
#
# data['next_mid_price'] = data['mid_price'].shift(-1)
# data.dropna(inplace=True)
# data['price_movement'] = np.where(data['next_mid_price'] > data['mid_price'], 1, 0)
# data.drop(['next_mid_price'], axis=1, inplace=True)

# Read the first 70% of the LOBs csv files from the CSV_10 folder
print("Reading the first 70% of the LOBs csv files from the CSV_10 folder...")
folder_path = "CSV_10"
# get csv file from the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# sort the csv files
csv_files.sort()
num_files_to_read = int(len(csv_files) * percentage)
# select the first 70% of the csv files
files_to_read = csv_files[:num_files_to_read]

# read the csv files into a list of dataframes
dataframes = []
for file in files_to_read:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# concatenate the dataframes into one dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# remove the "rising", "falling" and "stable" columns
combined_df.drop(['rising', 'falling', 'stable'], axis=1, inplace=True)
print("combined_df.shape: ", combined_df.shape)

# print(data.shape)
# print(data.columns)
data = combined_df.values
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


def create_LSTM_model():
    inputs = Input(shape=(batch_size, X_train.shape[1]))
    lstm = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    predictions = TimeDistributed(Dense(3, activation='softmax'))(lstm)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Create the model
print("Creating the model...")
model = create_LSTM_model()

# Train the model
print("Training the model...")
model.fit(X_train_batch, y_train_batch, epochs=epochs, batch_size=batch_size)

# Save the model
print("Saving the model...")
model.save('LSTM.h5')

# Plot the training loss and accuracy
print("Plotting the training loss and accuracy...")
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['accuracy'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='upper left')
plt.show()

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test_batch)

# # y_pred to categorical
# y_pred = np.argmax(y_pred, axis=2)
# print(y_pred)
# # y_test_batch to categorical
# y_test_batch = np.argmax(y_test_batch, axis=2)
# print(y_test_batch)

# y_pred to categorical
y_pred = np.argmax(y_pred, axis=2)
y_pred = to_categorical(y_pred)
# y_pred 3D to 2D
y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2]))
# y_test_batch 3D to 2D
y_test_batch = np.reshape(y_test_batch, (y_test_batch.shape[0] * y_test_batch.shape[1], y_test_batch.shape[2]))


print(classification_report(
    y_test_batch,
    y_pred,
    target_names=["Stable", "Rising", "Falling"],
    digits=5))
print("Accuracy: ", accuracy_score(y_test_batch, y_pred))





