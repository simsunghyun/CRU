import Train

if __name__ == '__main__':
    # Data Preprocessing
    # Example; Daily Delhi Climate Data
    
    data_name = 'DailyDelhiClimate.csv'
    #data_name = 'SP500.csv'
    
    data_type = 'univariate'
    time_window = 1
    forecasting_term = 1
    
    hid_dim = 2048
    out_dim = 1
    num_layers = 1
    epoch = 10000
    
    model = Train.ModelTrain(data_name, data_type, time_window, forecasting_term)
    m = model._fit(hid_dim, out_dim, num_layers, epoch)