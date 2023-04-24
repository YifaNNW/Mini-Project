import backtrader as bt
import numpy as np
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LSTMStrategy(bt.Strategy):
    def __init__(self):
        self.model = load_model('LSTM.h5')
        self.time_window = 128
        self.num_features = 6

    def next(self):
        # Get the data from the last 128 time steps
        input_data = np.array([self.data.time.get(size=self.time_window),
                              self.data.bid_weighted_average.get(size=self.time_window),
                              self.data.ask_weighted_average.get(size=self.time_window),
                              self.data.bid_ask_ratio.get(size=self.time_window),
                              self.data.mid_price.get(size=self.time_window)]).T
        # Scale the data
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
        # Reshape the data
        input_data = np.reshape(input_data, (1, self.time_window, self.num_features))
        # Predict the price movement
        y_pred = self.model.predict(input_data)
        # If y_pred is 1, then buy, if y_pred is 0, then sell
        if y_pred[-1] == 1:
            # Buy
            self.buy()
        else:
            # Sell
            self.sell()


class CustomPandasData(bt.feeds.PandasData):
    lines = ('time', 'bid_weighted_average', 'ask_weighted_average', 'bid_ask_ratio',
             'mid_price', 'price_movement',)
    params = (
        ('datetime', 0),
        ('time', -1),
        ('bid_weighted_average', -1),
        ('ask_weighted_average', -1),
        ('bid_ask_ratio', -1),
        ('mid_price', -1),
        ('price_movement', -1),
    )


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(LSTMStrategy)
    data_frame = pd.read_csv('2024-01-02LOBs.csv')

    # 将UNIX时间戳转换为pandas datetime类型
    data_frame['time'] = pd.to_datetime(data_frame['time'], unit='s')

    print(data_frame.head())
    data_feed = CustomPandasData(dataname=data_frame, datetime='time')

    cerebro.adddata(data_feed)
    cerebro.broker.setcash(1000000)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()



