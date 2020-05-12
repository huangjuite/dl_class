#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sn
import random
import copy
import matplotlib.pyplot as plt
import pygal
import cv2
from sklearn.metrics import confusion_matrix
from IPython.display import Image
from tqdm import trange, tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as trns
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# In[ ]:


method = "GRU" # "LSTM" or "GRU"
correlation_thres = 0.8
batch_size = 128
sequence_len=7
train_test_r = 0.7
learning_rate = 0.001
epoch = 600
evaluate_thres=0.5
visual_folder = "./visual_%s"%method


# In[ ]:


raw_data = pd.read_csv("covid_19.csv").drop([0,1])
infection_data = raw_data.drop(['Lat','Long'],axis=1)
country_names = raw_data.iloc[:,0]
position_data = raw_data.iloc[:,1:3].to_numpy().astype(np.float32)
infection_data = infection_data.iloc[:,1:].to_numpy().astype(np.float32)

infection_data = np.diff(infection_data,axis=1)
print("%d countries in data"%infection_data.shape[0])


# In[ ]:


core_mat = np.corrcoef(infection_data).astype(np.float32)
np.fill_diagonal(core_mat, 0)
core_max = np.max(core_mat,axis=1)
plt.title("maximum correlation coefficient with other countries")
plt.plot(core_max)
plt.savefig(visual_folder+'/correlation_coefficient.png', bbox_inches = "tight")
# plt.show()
core_mat = np.tril(core_mat)

plt.figure(figsize = (16,12))
plt.title("infections data correlation coefficient between 185 countries")
sn.heatmap(core_mat)
plt.savefig(visual_folder+'/correlation_matrix.png', bbox_inches = "tight")
# plt.show()


infection_inlier_data = infection_data[np.where(core_max>=correlation_thres)]
print("remove maximum correlation coefficient under %.3f"% correlation_thres)

print("%d countries left"%infection_inlier_data.shape[0])


# In[ ]:


def cut_sequence(data,sequence_len=14):
    sequence_data = []
    for i in range(data.shape[0]):
        for k in range(data.shape[1]-sequence_len):
            inputs = data[i][k:k+sequence_len]
            inputs = np.expand_dims(inputs,axis=1)
            label = torch.Tensor([1.0] if data[i][k+sequence_len-1] > data[i][k+sequence_len] else [0.0])
            sequence_data.append({'inputs':inputs,'label':label})
    return sequence_data


# In[ ]:


sequence_data = cut_sequence(infection_inlier_data,sequence_len)

class infectionDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.transform = trns.Compose([trns.ToTensor()])
        
    def __getitem__(self,index):
        inputs = self.data[index]['inputs']
        label = self.data[index]['label']
        
        return inputs, label
    
    def __len__(self):
        return len(self.data)


# In[ ]:


dataset = infectionDataset(sequence_data)
train_sz = int(len(dataset)*train_test_r)
test_sz = len(dataset)-train_sz
train_set, test_set = random_split(dataset,[train_sz,test_sz])

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_set,
                          batch_size=batch_size, 
                          shuffle=False,
                          num_workers=4)


# In[ ]:


class LSTMmodule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.W_ui = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # forget gate
        self.W_uf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # internal_gate
        self.W_ug = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
        # output gate
        self.W_uo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
         
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x):
        batch_size, seq_size, feature_size = x.size()
        h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        
        # iterate over the time steps
        for t in range(seq_size):
            # shape (batch, sequence, feature)
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.W_ui + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_uf + h_t @ self.W_hf + self.b_f)
            g_t = torch.tanh(x_t @ self.W_ug + h_t @ self.W_hg + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_uo + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
        
        return h_t


# In[ ]:


class GRUmodule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # update gate
        self.W_iu = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hu = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_u = nn.Parameter(torch.Tensor(hidden_size))
        
        # reset gate
        self.W_ir = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))
        
        # output gate
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        
         
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x):
        batch_size, seq_size, feature_size = x.size()
        h_t = torch.zeros(self.hidden_size).to(x.device)
        
        # iterate over the time steps
        for t in range(seq_size):
            # shape (batch, sequence, feature)
            x_t = x[:, t, :]
            u_t = torch.sigmoid(x_t @ self.W_iu + h_t @ self.W_hu + self.b_u)
            r_t = torch.sigmoid(x_t @ self.W_ir + h_t @ self.W_hr + self.b_r)
            
            o_t = torch.tanh(x_t @ self.W_io + ( r_t * h_t ) @ self.W_ho + self.b_o)
            
            h_t = (1-u_t) * h_t + u_t * o_t
        
        return h_t


# In[ ]:


def evaluate(model,dataloader):
    model.eval()
    error = 0
    total = len(dataloader.dataset)
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        y_hat = outputs.cpu().detach().numpy()
        y_hat = np.where(y_hat>evaluate_thres,1,0)
        y = labels.cpu().detach().numpy()
        error += np.count_nonzero(y-y_hat)
    model.train()
    return (1-(error/total))


# In[ ]:


hidden_size = 1
input_size = 1

if method == "LSTM":
    module = LSTMmodule(input_size, hidden_size)
elif method=="GRU":
    module = GRUmodule(input_size, hidden_size)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
module.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(module.parameters(), lr=learning_rate)

losses = []
train_acc = []
test_acc = []

train_acc.append(evaluate(module,train_loader))
test_acc.append(evaluate(module,test_loader))

t = trange(epoch)
for e in t:
    loss_log = []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = module.forward(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    train_acc.append(evaluate(module,train_loader))
    test_acc.append(evaluate(module,test_loader))
    losses.append(np.mean(loss_log))
    t.set_description("train_loss:%.4f, train_acc:%.4f, test_acc:%.4f"%(losses[-1],train_acc[-1],test_acc[-1]))


# In[ ]:


plt.title("train loss")
plt.xlabel("epochs")
plt.ylabel("average loss")
plt.plot(losses)
plt.savefig(visual_folder+'/train_loss.png', bbox_inches = "tight")
# plt.show()

plt.title("train accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(train_acc)
plt.savefig(visual_folder+'/train_accuracy.png', bbox_inches = "tight")
# plt.show()

plt.title("test accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(test_acc)
plt.savefig(visual_folder+'/test_accuracy.png', bbox_inches = "tight")
# plt.show()


# In[ ]:


COUNTRIES = {v: k for k, v in pygal.maps.world.COUNTRIES.items()}
M_COUNTRIES = {"Bolivia":"bo", 
               "Brunei":"bn", 
               "Congo (Brazzaville)":"cg", 
               "Congo (Kinshasa)":"cd", 
               "Czechia":"cz", 
               "Dominica":"do",
               "Eswatini":"sz",
               "Holy See":"va",
               "Iran":"ir",
               "Korea, South":"kr",
               "Laos":"la",
               "Libya":"ly",
               "Moldova":"md",
               "North Macedonia":"mk",
               "Russia":"ru",
               "Syria":"sy",
               "Taiwan*":"tw",
               "Tanzania":"tz",
               "US":"us",
               "Venezuela":"ve",
               "Vietnam":"vn",
              }
COUNTRIES.update(M_COUNTRIES)


# In[ ]:


test_data = infection_data[:,-14:]
test_data = np.expand_dims(test_data,axis=2)
test_data = torch.Tensor(test_data).to(device)
test_output = module(test_data)


# In[ ]:


accending = []
decending = []
print("missing : ")
for i in range(len(country_names)):
    try:
        if test_output[i][0] > evaluate_thres:
            accending.append(COUNTRIES[country_names.iloc[i]])
        else:
            decending.append(COUNTRIES[country_names.iloc[i]])
    except:
        print("    %s"%country_names.iloc[i])

worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'infections prediction'
worldmap_chart.add('accending', accending)
worldmap_chart.add('deccending', decending)
worldmap_chart.add('no data', COUNTRIES.values())

worldmap_chart.render_to_png(visual_folder+'/map.png')
# Image(filename=visual_folder+'/map.png')


# In[ ]:




