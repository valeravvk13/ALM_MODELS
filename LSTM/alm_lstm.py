import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split  

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import itertools

import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go

import torch 
import torch.nn as nn
import torch.nn.functional as F




class DataLoader():

    
    def __init__(self, 
                 data, 
                 categorical_features_names=[], 
                 continuous_features_names=[],
                 target_name=None,
                 seq_lenght=1
                ):
        
        self.data = data.copy()
        
        self.categorical_features = self.data[categorical_features_names]
        self.continuous_features = self.data[continuous_features_names]
        
        if target_name is not None:
            self.y = np.array(self.data[target_name].values).reshape(-1, 1)
            
        self.seq_lenght = seq_lenght

        
    def categorical_features_embedding(self):
        
        if self.categorical_features.shape[1] > 0:
        
            cat_feat = torch.tensor(np.array(self.categorical_features), dtype=torch.int64)

            cat_dims = [len(self.data[column].unique()) for column in self.categorical_features.columns]

            embedding_dims=[(dim_, min(50, (dim_ + 1) // 2)) for dim_ in cat_dims]

            embed_representation = nn.ModuleList([nn.Embedding(inp + 1, out) for inp, out in embedding_dims])

            embedding_val=[]
            for i, embed in enumerate(embed_representation):    
                embedding_val.append(embed(cat_feat[:, i]))

            cat_feat = torch.cat(embedding_val, 1)

            self.categorical_features = cat_feat.detach().numpy()
            return self.categorical_features
    
    
    def categorical_features_scale(self, categorical_features=None, 
                                   Scaler=MinMaxScaler(feature_range=(0, 1)) 
                                  ):
        
        if self.categorical_features.shape[1] > 0:
            #add some functions for categorical_features
            if categorical_features is None:
                categorical_features = self.categorical_features 

            self.__cat_Scaler = Scaler
            self.categorical_features = self.__cat_Scaler.fit_transform(categorical_features)
            return self.categorical_features
    
    
    def continuous_features_scale(self, continuous_features=None, 
                                  Scaler=MinMaxScaler(feature_range=(0, 1)) 
                                 ):
        #add some functions for continuous_features
        if continuous_features is None:
            continuous_features = self.continuous_features 
            
        self.__cont_Scaler = Scaler
        self.continuous_features = self.__cont_Scaler.fit_transform(continuous_features)
        return self.continuous_features
    
    
    def y_scale(self, y=None, Scaler=MinMaxScaler(feature_range=(0, 1)) ):
        if y is not None:
            self.y = y
        self.__y_Scaler = Scaler
        self.y = self.__y_Scaler.fit_transform(self.y)
        return self.y
        
        
    def get_Scaler(self, name='target'):
        
        if name == 'target':
            return self.__y_Scaler
        if name == 'continuous_features':
            return self.__cont_Scaler
        if name == 'categorical_features':
            return self.__cat_Scaler
        
    
    def prepare_sequence(self, X, Y=None):
    
        if Y is not None:
            x, y = [], []
            for i in range(len(np.array(X)) - self.seq_lenght):
                x.append(X[i:(i + self.seq_lenght)])
                y.append(Y[i + self.seq_lenght - 1])
            return np.array(x), np.array(y)
            
        else:
            x = []
            for i in range(len(np.array(X)) - self.seq_lenght):
                x.append(X[i:(i + self.seq_lenght)])
            return np.array(x)
        
    
    def sequence_features(self):
        if self.categorical_features.shape[1] > 0:
            self.categorical_features = self.prepare_sequence(self.categorical_features)
        self.continuous_features, self.y = self.prepare_sequence(self.continuous_features, self.y)
        
               
    def _torch_tensor(self, features, dtype=torch.float):
        return torch.tensor(features, dtype=dtype)
    
    
    def transform(self):
        
        _ = self.categorical_features_embedding()
        _ = self.categorical_features_scale()
        self.categorical_features = self.prepare_sequence(self.categorical_features)
    
        _ = self.y_scale()
        
        _ = self.continuous_features
        _ = self.continuous_features_scale()
        self.continuous_features, self.y = self.prepare_sequence(self.continuous_features, self.y)    
        
        torch_categorical_features = self._torch_tensor(self.categorical_features)
        torch_continuous_features = self._torch_tensor(self.continuous_features)
        torch_y = self._torch_tensor(self.y)
        
        return torch_categorical_features, torch_continuous_features, torch_y
        
        
        
torch.manual_seed(13)        
             
class LSTM_ts(nn.Module):
    
    def __init__(self, 
                 seq_lenght=7, 
                 output_size=1, 
                 embed_dropout=0.2, 
                 **lstm_kwargs, 
                ):
        super(LSTM_ts, self).__init__()
       
        self.input_size = lstm_kwargs['input_size']
        self.hidden_size = lstm_kwargs['hidden_size']
        self.num_layers =lstm_kwargs['num_layers']
        
        self.embed_drop = nn.Dropout(embed_dropout)
        self.bn_cont = nn.BatchNorm1d(seq_lenght)

        self.lstm = nn.LSTM(**lstm_kwargs, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, output_size)
        
     
    def forward(self, x_cont, x_cat=None):
        
        x = self.bn_cont(x_cont) #batchnorm continues features
        if x_cat is not None:
            x_cat = self.embed_drop(x_cat)  #dropout in embeddings
            x = torch.autograd.Variable(torch.cat([x, x_cat], axis=2)) 
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        out = self.fc(out[:, -1, :])
        return out
 
    