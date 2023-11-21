import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

device = 'cpu'

df = pd.read_csv('../Data/Raw/USTotalPrivate.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[(df['DATE'] < '2020-04-01') | (df['DATE'] > '2020-08-01')  ]
df_diff = df.diff(axis = 0)
df_diff['DATE'] = df['DATE']
df_diff
df_diff.rename(columns = {'USPRIV' : 'Total_priv', 'DATE': 'Month'}, inplace = True)
df_diff = df_diff.iloc[1:,:]
df = df_diff


from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    
    df.set_index('Month', inplace=True)
    
    for i in range(1, n_steps + 1):
        df[f'Total_priv(t-{i})'] = df['Total_priv'].shift(i)
        
    df.dropna(inplace=True)
    
    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(df, lookback)

shifted_df_as_np = shifted_df.to_numpy()

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]
y_dir = np.sign(y)
y_dir[y_dir == -1] = 0
X = dc(np.flip(X, axis=1)) # because we want to start from the earliest time)


split_index = int(len(X) * 0.8)
# split_index
X_train = X[:split_index]
X_test = X[split_index:]

y_train = y_dir[:split_index]
y_test = y_dir[split_index:]

num_classes = 2
X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

X_train = torch.tensor(X_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.long)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.long)

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        # or:
        #out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        out = self.softmax(out)
        return out
    

num_epochs = 50
input_size = 1
hidden_size = 64
num_layers = 8

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

learning_rate = 0.002
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

all_train_loss= []
all_test_loss = []
def train_one_epoch():
    model.train(True)
    if epoch % 10 == 9:
        print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # print(y_batch)
        
        output = model(x_batch)
        # print(output)
        # y_batch = y_batch.long()
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch_index % 20 == 19:  # print every 20 batches
        #     avg_loss_across_batches = running_loss / (20 * batch_size)
        #     print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
        #                                             avg_loss_across_batches))
        #     running_loss = 0.0
    # print()
    all_train_loss.append(running_loss / (len(train_loader) * batch_size))

def validate_one_epoch():
    model.train(False) #evaluation mode
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # x_batch = x_batch.float()
        # y_batch = y_batch.long()
        
        with torch.no_grad(): #not calculating gradients because we're not updating model
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    all_test_loss.append(running_loss / (len(test_loader) * batch_size))
    # avg_loss_across_batches = running_loss / (len(test_loader) * batch_size)
    
    # print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    # print('***************************************************')
    # print()

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()


with torch.no_grad():
    predicted_train = model(X_train.to(device)).to('cpu').numpy()

# print(predicted_train)
max_ind_train = np.argmax(predicted_train, axis = 1)
# print(max_ind_train)
error = np.sum(abs(y_train.numpy() - max_ind_train)) / len(max_ind_train)
print(f"training data error is {error}")

with torch.no_grad():
    predicted_test = model(X_test.to(device)).to('cpu').numpy()

max_ind_test = np.argmax(predicted_test, axis = 1)
error = np.sum(abs(y_test.numpy() - max_ind_test)) / len(max_ind_test)
print(f"test data error is {error}")

plt.plot(all_train_loss,  label='Train loss')
plt.plot(all_test_loss,  label='Test loss')
plt.legend()
plt.show()

# batch= next(iter(train_loader))
# X_batch, y_batch = batch[0].to(device), batch[1].to(device)
# # print("x batch type")
# # print(X_batch.dtype)
# # print(y_batch.shape)
# output = model(X_batch)
# print(output)
# # print(output.shape)

# loss_function = nn.NLLLoss()
# # print(y_batch)
# y_batch = y_batch.long()
# print(y_batch)
# # print('y batch data type')
# # print(y_batch.dtype)
# loss = loss_function(output, y_batch)
# print(loss)