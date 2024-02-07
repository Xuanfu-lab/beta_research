import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


data = pd.read_csv("data/cleaned_data.csv")
data['date'] = pd.to_datetime(data['date'])
selected_features = ['log_size', 'log_pcf', 'log_to', 'vol', 'ols_3m_d', 'ols_1y_d', 'ols_5y_m']
test_time_start = data['date'].max() - pd.DateOffset(years=5) # last 5 years as test
val_time_start = test_time_start - pd.DateOffset(years=2) # last 2 years as val
data_test = data[data['date'] >= test_time_start]
data_val = data[(data['date'] >= val_time_start) & (data['date'] < test_time_start)]
data_train = data[data['date'] < val_time_start]


# Data Preparation Pipeline
def lstm_data_pipeline(data: pd.DataFrame, T = 12):
    input_sequences = []
    targets = []
    for ticker in data['ticker'].unique():
        print(ticker)
        ticker_data = data[data['ticker'] == ticker].sort_values('date')
        for i in range(len(ticker_data) - T):
            sequence = ticker_data.iloc[i:i+T][selected_features].values
            target = ticker_data.iloc[i+T]['f_ols_1y_d']
            
            input_sequences.append(sequence)
            targets.append(target)
    
    input_sequences = np.array(input_sequences)
    input_tensor = torch.tensor(input_sequences)
    target_tensor = torch.tensor(targets)

    means = input_tensor.mean(dim=[0, 1])
    stds = input_tensor.std(dim=[0, 1])
    input_tensor = (input_tensor - means) / stds # standardize input features
    
    return input_tensor, target_tensor



X_train, y_train = lstm_data_pipeline(data_train)
X_val, y_val = lstm_data_pipeline(data_val)
X_test, y_test = lstm_data_pipeline(data_test)
torch.save(X_train, 'data/X_train.pt')
torch.save(X_val, 'data/X_val.pt')
torch.save(X_test, 'data/X_test.pt')
torch.save(y_train, 'data/y_train.pt')
torch.save(y_val, 'data/y_val.pt')
torch.save(y_test, 'data/y_test.pt')