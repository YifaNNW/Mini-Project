import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from statsmodels.tsa.stattools import adfuller


# draw the time series
def draw_ts(ts):
    plt.plot(ts)
    plt.show()


# test the stationarity of the time series
def mytest_stationarity(timeseries):
    dftest = adfuller(timeseries['#Passengers'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


if __name__ == '__main__':
    # Load the data
    X = pd.read_csv('AirPassengers.csv', encoding='utf-8', index_col='Month')


    # # draw the time series
    # draw_ts(X)
    # # First order differential
    # X_diff = X.diff(periods=1).dropna()
    # draw_ts(X_diff)
    # # Second order differential
    # X_diff2 = X_diff.diff(periods=1)
    # draw_ts(X_diff2)

    # # test the stationarity of the time series
    # print(mytest_stationarity(X))
    # # p-value                          0.991880
    # print(mytest_stationarity(X_diff))
    # # p-value                          0.054213

    ts_log = np.log(X)
    # draw_ts(ts_log)

    ts_log_diff = ts_log.diff(periods=1).dropna()
    # draw_ts(ts_log_diff)

    ts_log_diff_2 = ts_log_diff.diff(periods=1).dropna()
    # draw_ts(ts_log_diff_2)

    # print(mytest_stationarity(ts_log_diff_2))
    # p-value                        7.419305e-13
    # It is smoothed time series

    # # draw the ACF and PACF
    # from statsmodels.tsa.stattools import acf, pacf
    # lag_acf = acf(ts_log_diff_2, nlags=20)
    # lag_pacf = pacf(ts_log_diff_2, nlags=20, method='ols')
    #
    # # Plot ACF:
    # plt.subplot(121)
    # plt.plot(lag_acf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff_2)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff_2)), linestyle='--', color='gray')
    # plt.title('Autocorrelation Function')
    #
    # # Plot PACF:
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff_2)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff_2)), linestyle='--', color='gray')
    # plt.title('Partial Autocorrelation Function')
    # plt.tight_layout()
    # plt.show()
    # # The ACF and PACF curves show that the time series is a ARIMA(1,1,1) model

    # Build the ARIMA model
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(X, order=(1, 1, 1))
    results_ARIMA = model.fit()

    # # draw the residual
    # residuals = pd.DataFrame(results_ARIMA.resid)
    # from statsmodels.graphics.api import qqplot
    # fig = qqplot(residuals, line='q', fit=True)
    # plt.show()
    # # The residual is normal distribution

    # draw the prediction
    pred = results_ARIMA.predict(start=1, end=len(X)+10)

    # draw the prediction and the original time series in the same figure
    # Convert the horizontal coordinates to the same

    pred.index = pd.date_range(start='1949-01-01', periods=len(pred), freq='MS')
    X.index = pd.date_range(start='1949-01-01', periods=len(X), freq='MS')
    X.plot(label='original')
    pred.plot(label='prediction')
    plt.legend(loc='best')
    plt.show()

    ##TODO: try SARIMA model
    

