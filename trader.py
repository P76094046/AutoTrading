# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:22:11 2021

@author: P76094046
"""


# You can write code above the if-main block.
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import csv

class dataset(torch.utils.data.Dataset):     # for training set
    def __init__(self, data, label):
        # preporcssing
        self.data = data
        self.label = label
    def __len__(self):  
        return len(self.data) - num_4_pred - num_pred + 1
     
    def __getitem__(self, index):
        stock = self.data[index : index + num_4_pred]
        lb = self.label[index + num_4_pred : index + num_4_pred + num_pred]
        stock = torch.from_numpy(stock).type(torch.Tensor)
        lb = torch.from_numpy(lb).type(torch.Tensor)
        return stock, lb

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, None)
        out = self.fc(hn[1]) 
        # print(hn.shape)
        # print(out)
        return out       



if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # path = 'C://Users/P76094046/Desktop/DSAI/HW2/'
    column_names = ['open','high','low','close']
    training = pd.read_csv(args.training, names = column_names)
    testing = pd.read_csv(args.testing, names = column_names)
    n = len(training)
    data = pd.concat([training, testing])
    label = np.array(data['open'].values)
    
    num_4_pred = 14 ## number of days for predict 
    num_pred = 1
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.values)
    training = data[0: n]
    training_label = label[0:n]
    testing = data[n:]
    testing_label = label[n:]
    
    batch_size = 32
    num_epochs = 500 #n_iters / (len(train_X) / batch_size)
    
    dataset = dataset(training, training_label)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,#dataset 
                                               batch_size=batch_size, 
                                               shuffle=False)
    input_dim = 4
    hidden_dim = 32
    num_layers = 2 
    output_dim = 1
    
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.MSELoss()
    
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    # print(model)
    # print(len(list(model.parameters())))
    # for i in range(len(list(model.parameters()))):
    #   print(list(model.parameters())[i].size())
    
    # Train model 
    for t in range(num_epochs):
        for num , data in enumerate(train_loader, 0):
            input_data, label = data
            print(input_data.shape)
            pred = model(input_data)
            # print(pred)
            loss = loss_fn(pred, label)
            print('epcoh %d'%t, loss.item())
            # hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
     
    # Predict
    
    data = np.concatenate((training, testing))
    m = len(data)
    test_data = []
    for i in range(n - num_4_pred + 1, m - num_4_pred  + 1):
        temp = []
        for j in range(0, num_4_pred):
            temp.append(data[i + j])
        test_data.append(temp)
    
    pred_list = []
    for data in test_data:
        data = np.array(data)
        data = np.reshape(data, (-1, 14, 4)) 
        data= torch.from_numpy(data).type(torch.Tensor)
        pred = model(data)
        print(pred)
        pred_list.append(pred)   
        
    # Making decisions
    # Initialize
    State = 0
    action = 0
    action_list = []
    action = 1      #一開始先買
    temp = pred_list[0]
    State = State + action
    action_list.append(action)
    for i in range(1, len(pred_list)):
        if (pred_list[i] > temp):     # 比買價高
            if State > 0:       # 如果有股票 就賣
                action = -1
                State = State + action
                temp = 0
                action_list.append(action)
                print('A', pred_list[i], action)
            else:              # 如果沒股票
                if (pred_list[i] < pred_list[i-1]):     # 且比前一天低
                    action = 1
                    temp = pred_list[i]
                    State = State + action
                    action_list.append(action)
                    print('B', pred_list[i],action)
                else:         # 比前一天高
                    action = 0
                    State = State + action
                    action_list.append(action)
                    print('C', pred_list[i],action)
                
        else:                         # 比買價低
            if State > 0:             # 如果有股票 不動作
                action = 0
                State = State + action
                action_list.append(action)
                print('D', pred_list[i],action)
            else:                     # 如果沒股票 買
                action = 1
                temp = pred_list[i]
                State = State + action
                action_list.append(action)
                print('E', pred_list[i],action)
            
    # Output
    with open(args.output, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(action_list) -1):
            writer.writerow([action_list[i]])