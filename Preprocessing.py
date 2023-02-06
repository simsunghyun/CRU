import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class _Preprocessing():
    
    def __init__(self, data_name, data_type, time_window, forecasting_term, train_ratio = 0.7):
        
        # data_type = univariate time-series data, multivariate time-series data
        # time_window = input data length 
        # forecasting_term = forecasting point (ex. 1-step ahead, 3-step ahead)
        # train_ratio = train / test ratio  
        
        self.dt = pd.read_csv('data/' + data_name).iloc[:,1:].astype('float')
        self.data_type = data_type 
        self.time_window = time_window
        self.forecasting_term = forecasting_term
        self.train_ratio = train_ratio
        self.scaler = []
    
    def preprocessing(self):
        
        # Data Normalization
        for i in range(len(self.dt.columns)):
            self.scaler.append(MinMaxScaler())
                
        for i in range(len(self.dt.columns)):    
            self.scaler[i].fit(self.dt.iloc[:,i].values.reshape(-1,1))
                
        for i in range(len(self.dt.columns)):
            self.dt.iloc[:,i] = self.scaler[i].transform(self.dt.iloc[:,i].values.reshape(-1,1))
                
        # Train Test Seperation
        len_train = int(len(self.dt)*self.train_ratio)
        
        train = self.dt.iloc[:len_train,:]
        test = self.dt.iloc[len_train:,:]
        
        trainX, testX, trainY, testY = [], [], [], []
        
        for i in range(self.time_window, len(train)-self.forecasting_term):
            trainX.append(train.iloc[(i-self.time_window):i,:].values)
            trainY.append(train.iloc[i+self.forecasting_term,:].values)

        for i in range(self.time_window, len(test)-self.forecasting_term):
            testX.append(test.iloc[(i-self.time_window):i,:].values)
            testY.append(test.iloc[i+self.forecasting_term,:].values)
            
        trainX = torch.FloatTensor(trainX).cuda()
        trainY = torch.FloatTensor(trainY).cuda()
        
        testX = torch.FloatTensor(testX).cuda()
        testY = torch.FloatTensor(testY).cuda()
        
        return trainX, trainY, testX, testY, self.scaler
        
        