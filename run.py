import torch
import pandas as pd
import numpy as np
from importlib import import_module
from util import data_prepare,build_iterator,data_prepare_advanced,build_iterator_advanced
from train_eval_snr import train

if __name__ == '__main__':
    model_name = 'snr_trans'
    x = import_module('Models.'+model_name)
    config = x.Config('./data/census/')

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    #train_set,test_set,dev_set = data_prepare(config)
    train_ind = pd.read_csv("./data/census/train_ind.csv")
    train_label = pd.read_csv("./data/census/train_label.csv")
    train_set = pd.concat([train_ind,train_label],axis = 1)

    test_ind = pd.read_csv("./data/census/test_ind.csv")
    test_label = pd.read_csv("./data/census/test_label.csv")
    test_set = pd.concat([test_ind,test_label],axis = 1)

    dev_ind = pd.read_csv("./data/census/val_ind.csv")
    dev_label = pd.read_csv("./data/census/val_label.csv")
    dev_set = pd.concat([dev_ind,dev_label],axis = 1)
    
    train_iter = build_iterator(train_set, config)
        
    dev_iter = build_iterator(dev_set, config)
    test_iter = build_iterator(test_set, config)

    model = x.Model(config).to(config.device)

    print(model.parameters)
    train(config,model,train_iter,dev_iter,test_iter)
