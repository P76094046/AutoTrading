# DSAI HW2


## Data preprocessing

- 將數據歸一化至 (-1, 1) 區間
- 連接 training data 和 testing data，並取開盤價，以便放入模型預測
    - 不是使用整個連接後的資料做預測，而是指"用 training data 的最後 14 筆，預測 testing data 的第 1 筆"
    - 下一次則是 "使用 training data 的最後 13 筆 和 第 1 筆 testing data 預測 testing data 的第 2 筆"


## Model : LSTM

- 使用基礎的LSTM模型預測股價後再利用預測的股價做決策
- 用 14 天的資料預測 1 天
- 使用 Pytorch 建立模型
- Parameters:
    - Epoch = 500, 
    - Batch size = 32, 
    - Learning rate = 0.01, 
    - Optimizer = Adam


## Decision

- 第一天都先買股票(Buy)，並記錄買價。
- 如果後面的天數比買價高的話，若手上有股票就賣掉(Sell)；如果比買價低，就不賣(Hold)。
- 手上沒有股票的時候，若偵測到今天預測的股價比昨天預測的低，則決定買。
