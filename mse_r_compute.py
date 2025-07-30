# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:26:19 2024

@author: haizhouy
"""
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


def mse_r_compute(Device_data,Device_data_predict,train_idx,val_idx,test_idx):
    r_train = np.empty(0)
    r_val = np.empty(0)
    r_test = np.empty(0)

    for idx in range (Device_data.shape[1]+1):
        if idx == Device_data.shape[1]:
            r_train = np.append(r_train,stats.pearsonr(np.prod(Device_data_predict[train_idx],axis=1)/100,np.prod(Device_data[train_idx],axis=1)/100).statistic)
            r_val = np.append(r_val,stats.pearsonr(np.prod(Device_data_predict[val_idx],axis=1)/100,np.prod(Device_data[val_idx],axis=1)/100).statistic)
            r_test = np.append(r_test,stats.pearsonr(np.prod(Device_data_predict[test_idx],axis=1)/100,np.prod(Device_data[test_idx],axis=1)/100).statistic)
        else:      
            r_train = np.append(r_train,stats.pearsonr(Device_data_predict[train_idx,idx],Device_data[train_idx,idx]).statistic)
            r_val = np.append(r_val,stats.pearsonr(Device_data_predict[val_idx,idx],Device_data[val_idx,idx]).statistic)
            r_test = np.append(r_test,stats.pearsonr(Device_data_predict[test_idx,idx],Device_data[test_idx,idx]).statistic)

    mse_train = np.append(np.mean((Device_data_predict[train_idx]-Device_data[train_idx])** 2,axis=0),np.mean((np.prod(Device_data_predict[train_idx],axis=1)/100-np.prod(Device_data[train_idx],axis=1)/100)**2,axis=0))
    mse_val = np.append(np.mean((Device_data_predict[val_idx]-Device_data[val_idx])** 2,axis=0),np.mean((np.prod(Device_data_predict[val_idx],axis=1)/100-np.prod(Device_data[val_idx],axis=1)/100)**2,axis=0))
    mse_test = np.append(np.mean((Device_data_predict[test_idx]-Device_data[test_idx])** 2,axis=0),np.mean((np.prod(Device_data_predict[test_idx],axis=1)/100-np.prod(Device_data[test_idx],axis=1)/100)**2,axis=0))

    mse_summary = np.column_stack((mse_train, mse_val, mse_test))
    r_summary = np.column_stack((r_train, r_val, r_test))
    
    return mse_summary,r_summary