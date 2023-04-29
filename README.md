# Mini-Project
### 2023/3/11
Please check the Part1 notebook, I added the features we mentioned in class. 

### 2023/4/18
Processing of ohlc dataset, add the change trend (rising, falling and stable) of close price.
- week22_ohlc_processing.ipynb
- ohlc_trend.csv

### 2023/04/21
Downsample all LOBs datasets every ten seconds.
- week23.ipynb
- down sample datasets: CSV_10.zip

### 2023/04/29
Extract more features based on existing results:    
| time   | bid_weighted_average | bid_high | bid_volumn | ask_weighted_average | ask_low | ask_volumn | mid_price | spread | bid_ask_ratio | Trend | rising | falling | stable |
|--------|---------------------|----------|------------|---------------------|---------|------------|-----------|--------|---------------|-------|--------|---------|--------|
| 18.941 | 203.3               | 253.02   | 45.52      | 411.28              | 259.14  | 19.49      | 256.08    | 6.13   | 2.36          | 1     | 1      | 0       | 0      |
| 28.985 | 201.96              | 253      | 48.56      | 351.66              | 258.1   | 17.54      | 255.55    | 5.1    | 2.86          | 0     | 0      | 0       | 1      |
| 39.029 | 208.45              | 253.1    | 53.54      | 369.24              | 257.27  | 20.55      | 255.19    | 4.16   | 2.67          | 0     | 0      | 0       | 1      |
| 49.073 | 227.44              | 257.12   | 46.51      | 403.56              | 260.47  | 17.8       | 258.79    | 3.35   | 2.63          | 1     | 1      | 0       | 0      |
| 59.365 | 225.41              | 258.25   | 40         | 336.49              | 261.04  | 19.48      | 259.64    | 2.79   | 2.09          | 0     | 0      | 0       | 1      |

- time: The time of the observation, and the interval sets to 10s.
- bid_weighted_average: The weighted average bid price.  
- bid_high: The highest price a buyer is willing to pay for the stock.  
- bid_volumn: The total number of shares that buyers are willing to buy at the corresponding bid price.  
- ask_weighted_average: The weighted average ask price.  
- ask_low: The lowest price a seller is willing to accept for the stock.  
- ask_volumn: The total number of shares that sellers are willing to sell at the corresponding ask price.   
- mid_price: The average price between the highest bid price and lowest ask price.   
- spread: The difference between the highest bid price and lowest ask price.   
- bid_ask_ratio: The ratio of bid volume to ask volume.   
- Trend: Trend: The direction of the price trend. A value of 1 indicates the price is rising, while a value of 2 indicates the price is falling. A value of 0 indicates the price is stable.   
- rising: A binary variable indicating whether the price trend is rising (1) or not (0).   
- falling: A binary variable indicating whether the price trend is falling (1) or not (0).   
- stable: A binary variable indicating whether the price trend is stable (1) or not (0).   
