from tkinter import HIDDEN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 数据准备
df_app = pd.read_csv('AppRNN.csv', index_col = 'Date', parse_dates = ['Date'])
Train = df_app[:'2020-09-30'].iloc[:, 0:1].values
Test = df_app['2020-10-01':].iloc[:, 0:1].values

from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
Train = Scaler.fit_transform(Train)
Test = Scaler.transform(Test)

def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

seq_length = 4

x_train, y_train = sliding_windows(Train, seq_length)
x_test, y_test = sliding_windows(Test, seq_length)

trainX = Variable(torch.Tensor(np.array(x_train)))
trainY = Variable(torch.Tensor(np.array(y_train)))
testX = Variable(torch.Tensor(np.array(x_test)))
testY = Variable(torch.Tensor(np.array(y_test)))

# 模型构建
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):   
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 64
num_layers = 1
output_size = 1

# 模型训练
rnn = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.01)
num_epochs = 100
for epoch in range(num_epochs):
    outputs = rnn(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测
rnn.eval()
test_outputs = rnn(testX)
test_outputs = test_outputs.data.numpy()
test_outputs = Scaler.inverse_transform(test_outputs)
y_test_actual = Scaler.inverse_transform(y_test)
for i in range(len(y_test)):
    print(f"Date: {df_app['2020-10-01':].index[i+seq_length]}, Actual Activation: {y_test_actual[i][0]}, Predicted Activation: {test_outputs[i][0]}")

import matplotlib.pyplot as plt
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real Count') #真值
    plt.plot(predicted, color='blue',label='Predicted Count') #预测值
    plt.title('Flower App Activation Prediction') #图题
    plt.xlabel('Time') #X轴时间
    plt.ylabel('Flower App Activation Count') #Y轴激活数
    plt.legend() #图例
    plt.show() #绘图
    
plot_predictions(y_test_actual,test_outputs)