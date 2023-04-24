import pandas as pd
from keras.models import load_model
from LSTM import reshape_to_batches
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load the selected LOBs
print('Loading the LOBs...')
data_original = pd.read_csv('2024-01-02LOBs.csv')
data = data_original.values
X = data[:, :-1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

batch_size = 128
data_batch = reshape_to_batches(data, batch_size)

# Load the trained LSTM model
print("Loading the trained LSTM model...")
model = load_model('LSTM.h5')

# Set the initial capital
capital = 1000000

# Predict the mid_price movement for each batch
steps = len(data_batch) - 1
step = 0
trading_num = 10
shares_num = 100
list_buy_sell_index = []

for batch in data_batch:
    # Skip the last batch
    if step == steps:
        break
    # Reshape the batch
    batch = np.expand_dims(batch, axis=0)

    y_pred = model.predict(batch)

    # Using the last prediction as the signal for trading
    signal = y_pred[-1][-1][-1]
    # Get the mid_price of the last row
    mid_price = data_original['mid_price'].iloc[step * batch_size + batch_size - 1]

    # If the signal is 1, buy 10 shares
    if signal >= 0.5:
        print('Buy 10 shares at ', mid_price)
        shares_num += trading_num
        capital -= mid_price * trading_num
        list_buy_sell_index.append((step * batch_size + batch_size - 1, 'buy', mid_price))
    # If the signal is 0, sell 100 shares
    elif signal < 0.5 and shares_num >= trading_num:
        print('Sell 10 shares at ', mid_price)
        shares_num -= trading_num
        capital += mid_price * trading_num
        list_buy_sell_index.append((step * batch_size + batch_size - 1, 'sell', mid_price))
    else:
        print('Hold at ', mid_price)

    print('step: ', step, '/', steps)
    print('capital: ', capital)
    print('shares_num: ', shares_num)
    step += 1

# Plot the first 1000 mid_price and the buy/sell points
plt.plot(data_original['mid_price'].iloc[:1000], label='mid_price')
for index, buy_sell, mid_price in list_buy_sell_index:
    if index < 1000:
        if buy_sell == 'buy':
            plt.scatter(index, mid_price, c="green")
        else:
            plt.scatter(index, mid_price, c='red')

green_circle = mlines.Line2D([], [], color='green', marker='o', linestyle='None', label='buy')
red_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='sell')

plt.legend(handles=[green_circle, red_circle], labels=['buy', 'sell'])
plt.legend()

plt.show()










