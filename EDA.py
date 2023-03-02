import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read data
data_trades = pd.read_csv('HSBC_Examples/TstUoB_2024-01-02tapes.csv')
# print number of samples
print('Number of samples: ', len(data_trades))
# create a dataframe without the second column
data_trades = data_trades.drop(data_trades.columns[1], axis=1)

# create header for data
header = ['Day',  'TimeStamp', 'Price', 'Volume']
# set header
data_trades.columns = header

# data range
print(data_trades.head())
# print data covers span of time
print('Data covers span of time: ', data_trades['TimeStamp'].min(), ' to ', data_trades['TimeStamp'].max())
# print diff of prices
print('Diff of prices: ', data_trades['Price'].max() - data_trades['Price'].min())
# print diff of volumes
print('Diff of volumes: ', data_trades['Volume'].max() - data_trades['Volume'].min())

# print number of unique prices
print('unique values #####################')
print('Number of unique prices: ', len(data_trades['Price'].unique()))
# print number of unique volumes
print('Number of unique volumes: ', len(data_trades['Volume'].unique()))
# print number of unique timestamps
print('Number of unique timestamps: ', len(data_trades['TimeStamp'].unique()))

# group data by timestamp to calculate the average price and the total volume for each timestamp
data_trades = data_trades.groupby('TimeStamp').agg({'Price': 'mean', 'Volume': 'sum'})
print(data_trades.head())

# Plot the price and volume over time
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(data_trades['Price'])
plt.title('Price')
plt.xlim(0, 1000)
plt.ylim(270, 310)
plt.subplot(1, 2, 2)
plt.plot(data_trades['Volume'])
plt.title('Volume')
plt.xlim(0, 200)
plt.show()

# Using boxplot to visualize the distribution of prices and volumes
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.boxplot(data_trades['Price'])
plt.title('Price')
plt.subplot(1, 2, 2)
plt.boxplot(data_trades['Volume'])
plt.title('Volume')
plt.show()

# Using distplot to visualize the distribution of prices and volumes
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(data_trades['Price'], bins=100)
plt.title('Price')
plt.subplot(1, 2, 2)
sns.distplot(data_trades['Volume'], bins=100)
plt.title('Volume')
plt.show()
# the majority of the prices are between 280 and 290

# Using scatter plot to visualize the relationship between price and volume
plt.figure(figsize=(15, 5))
plt.scatter(data_trades['Price'], data_trades['Volume'])
plt.title('Price vs Volume')
plt.show()
















