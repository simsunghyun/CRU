import Preprocessing
import model.CRU as cru

from sklearn.metrics import mean_squared_error as mse
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim

class ModelTrain():
    
    def __init__(self, data_name, data_type, time_window, forecasting_term):

        self.in_dim = time_window
        pre = Preprocessing._Preprocessing(data_name, data_type, time_window, forecasting_term)
        self.trainX, self.trainY, self.testX, self.testY, self.scaler = pre.preprocessing()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
                    
    def _fit(self, hid_dim, out_dim, num_layers, epoch):
        
        m = cru._CRU(self.in_dim, hid_dim, out_dim, num_layers).to_device(self.device)
        crit = nn.MSELoss()
        para = list(m.parameters())
        optimizer = optim.Adam(para, 0.0001)
        
        for e in range(epoch):
            optimizer.zero_grad()
            out, _ = m(self.trainX)
            loss = crit(out, self.trainY.view(-1,1))
            loss.backward()
            optimizer.step()
        
            if (e+1) % 100 == 0:
                with torch.no_grad():
                    pred,_ = m(self.testX)
                    pred = pred.cpu().detach().numpy()
                    real = self.testY.cpu().numpy().reshape(-1,1)
                    for i in range(len(self.scaler)):
                        pred[i,:] = self.scaler[i].inverse_transform(pred[i,:].reshape(-1,1))
                        real[i,:] = self.scaler[i].inverse_transform(real[i,:].reshape(-1,1))
                        
                    test_mape = np.mean(abs((np.array(pred)-np.array(real))/np.array(pred)))*100
                    test_rmse = mse(pred, real)**0.5
                
                print('[Epoch: {}/{}] [Train Loss: {}] [Test RMSE: {}] [Test MAPE: {}]'.format(
                    e+1, epoch, str(loss.item())[:6], str(test_rmse)[:6], str(test_mape)[:6]))
        
        return m
                        
                        
                        
                        
                    