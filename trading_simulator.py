import pandas as pd
from keras.models import load_model
from LSTM import train_test_split, reshape_to_batches, _3d_to_2d, classification_result
import numpy as np

# Load the selected LOBs
data = pd.read_csv('2024-01-02LOBs.csv')
batch_size = 128
data = reshape_to_batches(data, batch_size)
# Predict the mid_price using LSTM
# Load the trained LSTM model

model = load_model('LSTM.h5')

# Set the initial capital
capital = 1000000

# Predict the mid_price movement
# First extract the first batch of data, add a batch each time and predict the mid_price movement
X_test = data[0, :, :]
for i in range(0, data.shape[0]):
    if i == 0:
        X_test = data[i, :, :]
    else:
        X_test = np.vstack((X_test, data[i, :, :]))
    y_pred = model.predict(X_test)
    # If y_pred is 1, then buy, if y_pred is 0, then sell
    if y_pred[-1] == 1:
        # Buy
        capital = capital - X_test[-1, 0]
    else:
        # Sell
        capital = capital + X_test[-1, 0]
    print(capital)

# Print the final capital
print(capital)










