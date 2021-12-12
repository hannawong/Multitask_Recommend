# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import loss_fn
import numpy as np

stat = open("data/census/feat_dict").read().split(" ")
FEAT_NUM = int(stat[0])+1
COL_NUM = int(stat[1])


class Config(object):
    def __init__(self, data_dir):
        self.model_name = 'mmoe'
        self.train_path = data_dir + 'census-income.data.gz'
        self.test_path = data_dir + 'census-income.test.gz'
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.dropout = 0.5
        self.learning_rate = 3e-5
        self.label_columns = ['income_50k', 'marital_stat']

        self.label_dict = [2, 2]
        self.num_feature = 0
        self.num_experts = 3
        self.num_tasks = 2
        self.units = 16
        self.hidden_units = 8
        self.embed_size = 10
        self.batch_size = 512
        self.field_size = 0
        self.towers_hidden = 16
        self.SB_hidden = 1024
        self.SB_output = 512
        self.num_epochs = 100
        self.loss_fn = loss_fn('binary')


class Transform_layer(nn.Module):
    def __init__(self, input_size, output_size, config): ## input_size:400,output_size:1024
        super(Transform_layer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.rand((1,), device=config.device), requires_grad=True)
        self.beta = 0.9
        self.gamma = -0.1
        self.eplison = 2

        w = torch.empty(input_size, config.num_experts,output_size, device=config.device)
        self.u = torch.nn.Parameter(torch.nn.init.uniform_(w, 0, 1),
                                    requires_grad=True)

        w = torch.empty(input_size,config.num_experts, output_size, device=config.device)
        self.w_params = torch.nn.Parameter(torch.nn.init.xavier_normal_(w),
                                           requires_grad=True)

    def forward(self, x):
        
        self.s = torch.sigmoid(torch.log(self.u) - torch.log(1 - self.u) + torch.log(self.alpha) / self.beta)
        
        self.s_ = self.s * (self.eplison - self.gamma) + self.gamma

        self.z_params = (self.s_ > 0).float() * self.s_
        self.z_params = (self.z_params > 1).float() + (self.z_params <= 1).float() * self.z_params

        output = self.z_params * self.w_params
        output = torch.einsum('ab,bnc -> anc', x, output)
        return output

class high_layers(nn.Module):

    def __init__(self,input_size,output_size,config):
        super(high_layers,self).__init__()
        self.alpha = torch.nn.Parameter(torch.rand((1,), device=config.device), requires_grad=True)
        self.beta = 0.9
        self.gamma = -0.1
        self.eplison = 2

        w = torch.empty(input_size, output_size, device=config.device)
        self.u = torch.nn.Parameter(torch.nn.init.uniform_(w, 0, 1),
                                    requires_grad=True)

        w = torch.empty(input_size, output_size, device=config.device)
        self.w_params = torch.nn.Parameter(torch.nn.init.xavier_normal_(w),
                                           requires_grad=True)
    def forward(self,x):
        self.s = torch.sigmoid(torch.log(self.u) - torch.log(1 - self.u) + torch.log(self.alpha) / self.beta)
        self.s_ = self.s * (self.eplison - self.gamma) + self.gamma

        self.z_params = (self.s_ > 0).float() * self.s_
        self.z_params = (self.z_params > 1).float() + (self.z_params <= 1).float() * self.z_params

        output = self.z_params * self.w_params
        output = torch.einsum('anc,cd -> and', x, output)
        return output


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        out = torch.sigmoid(out)
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # accept_unit = config.field_size*config.embed_size
        accept_unit = COL_NUM*config.embed_size ##[TODO]
        self.embedding = nn.Embedding(FEAT_NUM, config.embed_size)
        self.trans1 = Transform_layer(accept_unit, config.SB_hidden, config)
        self.trans2 = high_layers(config.SB_hidden,config.SB_output,config)

        self.fc_experts = nn.Linear(config.num_experts,1)
        self.relu = nn.ReLU()

        self.towers = nn.ModuleList([Tower(config.SB_output, 1, config.towers_hidden) for i in range(config.num_tasks)])

        self.lamdba = 1e-4

        # self.embedding_layer = nn.Embedding(config.num_feature,config.embed_size)

    def forward(self, x):

        x = self.embedding(x)
        self.field_size = x.shape[1]
        x = torch.cat([x[:,i,:] for i in range(0,self.field_size)],axis = 1) ## after embedding, [256, 400]:[batchsize, dim] 

        output = self.trans1(x)
        output = self.trans2(output)
        output = output.transpose(2,1)
        output = self.fc_experts(output)
        output = torch.squeeze(output)
        output = self.relu(output)

        final_outputs = [tower(output) for tower in self.towers]

        s1 = self.trans1.s_
        s2 = self.trans2.s_

        s1_prob = 1 - ((s1 < 0).sum(dim=-1) // s1.size(1))
        s2_prob = 1 - ((s2 < 0).sum(dim=-1) // s2.size(1))

        regul = self.lamdba * (s1_prob.sum() + s2_prob.sum())
        return final_outputs, regul

