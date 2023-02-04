import pandas as pd

class _Preprocessing():
    
    def __init__(self, data_name, data_type, time_window, forecasting_term, train_ratio = 0.7):
        
        # data_type = univariate time-series data, multivariate time-series data
        # time_window = input data length 
        # forecasting_term = forecasting point (ex. 1-step ahead, 3-step ahead)
        # train_ratio = train / test ratio 
        
        self.dt = pd.read_csv('data/' + data_name)
        self.data_type = data_type 
        self.time_window = time_window
        self.forecasting_term = forecasting_term
        self.train_ratio = train_ratio
        
    
    def preprocessing(self):
        
        self.X = self.dt.iloc[:,:-self.forecasting_term]
        self.Y = self.dt.iloc[:,self.forecasting_term:]
        
        
        
        