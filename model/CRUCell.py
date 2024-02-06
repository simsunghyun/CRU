import torch 
import torch.nn as nn
from torch.autograd import Variable
from statsmodels.tsa.seasonal import STL

import pandas as pd
import numpy as np


class _CRUCell(nn.Module):
    
    def __init__(self, in_dim, hid_dim, bias=True):
        super(_CRUCell, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.bias = bias
        
        self.wx_t = nn.Linear(self.in_dim, self.hid_dim*3, bias = self.bias)
        self.wx_s = nn.Linear(self.in_dim, self.hid_dim*3, bias = self.bias)
        self.wx_r = nn.Linear(self.in_dim, self.hid_dim, bias = self.bias)
        
        
        self.wh_t = nn.Linear(self.hid_dim, self.hid_dim*3, bias = self.bias)
        self.wh_s = nn.Linear(self.hid_dim, self.hid_dim*3, bias = self.bias)
        self.wh_r = nn.Linear(self.hid_dim, self.hid_dim, bias = self.bias)
        
        self.reset_parameters()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
            
    def forward(self, x, hid_state = None):
        
        lamda = 0.5
        x_t, x_s, x_r = self.ts_decompose(x)
        
        if hid_state is None:2
            hid_state = Variable(3, x.size(1), self.hid_dim).to(device)
        else:
            hid_state.cpu()
        
        with torch.no_grad():
            
            x_t = self.wx_t(x_t).to(self.device)
            x_s = self.wx_s(x_s).to(self.device)
            x_r = self.wx_r(x_r).to(self.device)
            
            h_t = self.wh_t(hid_state[0,:,:]).to(self.device)
            h_s = self.wh_s(hid_state[1,:,:]).to(self.device)
            
        x_autocor_t, x_cor_t, x_new_t = (x_t).chunk(3,1)
        x_autocor_s, x_cor_s, x_new_s = (x_s).chunk(3,1)
        h_autocor_t, h_cor_t, h_new_t = (h_t).chunk(3,1)
        h_autocor_s, h_cor_s, h_new_s = (h_s).chunk(3,1)
        
        autocor_t = torch.sigmoid(x_autocor_t + h_autocor_t)
        autocor_s = torch.sigmoid(x_autocor_s + h_autocor_s)
        
        cor_t = torch.sigmoid(x_cor_t + h_cor_s)
        cor_s = torch.sigmoid(x_cor_s + h_cor_t)
        
        new_t = lamda*torch.tanh(x_new_t + (autocor_t * h_new_t)) + (1-lamda)*torch.tanh(x_new_t + (cor_t * h_new_s))
        new_s = lamda*torch.tanh(x_new_s + (autocor_s * h_new_s)) + (1-lamda)*torch.tanh(x_new_s + (cor_s * h_new_t))
        
       
        with torch.no_grad():
            hid_state[0,:,:] = new_t
            hid_state[1,:,:] = new_s
            hid_state[2,:,:] = torch.tanh(x_r)
        
        return hid_state

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hid_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def ts_decompose(self, x):
        
        x = pd.DataFrame(x.cpu().numpy())
        
        dates = pd.date_range('1990-01-01', periods = len(x), freq='D')
        x.index = dates
        trend = []
        seasonal = []
        resid = []
        for i in range(len(x.columns)):
            stl = STL(x[i])
            res = stl.fit()
            trend.append(res.trend.values)
            seasonal.append(res.seasonal.values)
            resid.append(res.resid.values)
        
        trend = torch.FloatTensor(trend).permute(1,0).to(self.device)
        seasonal = torch.FloatTensor(seasonal).permute(1,0).to(self.device)
        resid = torch.FloatTensor(resid).permute(1,0).to(self.device)
        
        return trend, seasonal, resid