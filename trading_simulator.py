import pandas as pd
import os
from keras.models import load_model
from LSTM import reshape_to_batches
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Read the last 30% of the LOBs csv files from the CSV_101 folder
print('Loading the LOBs...')
folder_path = "CSV_10"
# get csv file from the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# sort the csv files
csv_files.sort()
# select the last 30% of the csv files
files_to_read = csv_files[int(len(csv_files) * 0.7):]
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

data_original = combined_df

# data_original = data_original.iloc[0:2000]

data = data_original.values
y = np.expand_dims(data[:, -1], axis=1)
X = data[:, :-1]

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

batch_size = 128
data_batch = reshape_to_batches(X, batch_size)

# Load the trained LSTM model
print("Loading the trained LSTM model...")
model = load_model('LSTM.h5')

# Set the initial capital
capital = 1000000

# Predict the mid_price movement for each batch
steps = len(data_batch) - 1
step = 0

shares_num = 100
list_buy_sell_index = []
last_mid_price = 0

for batch in data_batch:
    # Skip the last batch
    if step == steps:
        # save the last mid_price
        last_mid_price = data_original['mid_price'].iloc[(step - 1) * batch_size + batch_size - 1]
        break
    # Reshape the batch
    batch = np.expand_dims(batch, axis=0)

    y_pred = model.predict(batch)
    # Using the last prediction as the signal for trading
    y_pred = y_pred[-1]
    y_pred = np.argmax(y_pred, axis=1)
    signal = y_pred[-1]
    print('signal: ', signal)
    # Get the mid_price of the last row
    # mid_price = data_original['mid_price'].iloc[step * batch_size + batch_size - 1]
    ask_volume = data_original['ask_volumn'].iloc[step * batch_size + batch_size - 1]
    ask_low = data_original['ask_low'].iloc[step * batch_size + batch_size - 1]
    bid_volume = data_original['bid_volumn'].iloc[step * batch_size + batch_size - 1]
    bid_high = data_original['bid_high'].iloc[step * batch_size + batch_size - 1]
    # If the signal is 1, buy 10 shares
    if signal == 1 and capital >= ask_low * ask_volume:

        print('Buy {} shares at {}'.format(ask_volume, ask_low))
        shares_num += ask_volume
        capital -= ask_low * ask_volume
        list_buy_sell_index.append((step * batch_size + batch_size - 1, 'buy', ask_low))
    # If the signal is 0, sell 100 shares
    elif signal == 2 and shares_num >= bid_volume:
        print('Sell {} shares at {}'.format(bid_volume, bid_high) )
        shares_num -= bid_volume
        capital += bid_high * bid_volume
        list_buy_sell_index.append((step * batch_size + batch_size - 1, 'sell', bid_high))

        # # sell all shares
        # print('Sell all shares at ', mid_price)
        # capital += mid_price * shares_num
        # shares_num = 0
        # list_buy_sell_index.append((step * batch_size + batch_size - 1, 'sell', mid_price))

    else:
        print('Hold at ', ask_low)

    print('step: ', step, '/', steps)
    print('capital: ', capital)
    print('shares_num: ', shares_num)
    step += 1

# Calculate the final capital
final_capital = capital + last_mid_price * shares_num
print('final_capital: ', final_capital)
left = 10000
right = 10300
# Plot the 10000-10500 mid_price,ask_low,bid_high and the buy/sell points
plt.plot(data_original['mid_price'].iloc[left:right], label='mid_price')
plt.plot(data_original['ask_low'].iloc[left:right], label='ask_low')
plt.plot(data_original['bid_high'].iloc[left:right], label='bid_high')

# Plot the (10000,10500) buy/sell points
# Plot the (10000,10500) buy/sell points
for index, action, price in list_buy_sell_index:
    if left <= index < right:
        if action == 'buy':
            plt.plot(index - left, price, 'go')
        else:
            plt.plot(index - left, price, 'ro')

green_circle = mlines.Line2D([], [], color='green', marker='o', linestyle='None', label='buy')
red_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='sell')
mid_price_line = mlines.Line2D([], [], color='blue', label='mid_price')
bid_high_line = mlines.Line2D([], [], color='orange', label='bid_high')
ask_low_line = mlines.Line2D([], [], color='green', label='ask_low')

plt.legend(handles=[mid_price_line, bid_high_line, ask_low_line, green_circle, red_circle])

plt.show()










