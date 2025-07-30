# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:49:54 2025

@author: haizhouy
"""
import numpy as np
from compute_mean_std import propagated_pce_std
def ConfInteAccu(Device_data, Device_data_predict,Device_data_predict_std,train_idx,val_idx,test_idx):
    
    
    
    Device_data_lower_bound = Device_data_predict - 1.96 * Device_data_predict_std
    Device_data_upper_bound = Device_data_predict + 1.96 * Device_data_predict_std

    # Check if PCE falls within bounds
    Device_data_is_within_bounds = (Device_data >= Device_data_lower_bound) & (Device_data <= Device_data_upper_bound)
    
    Device_data_accuracy_percentage_train = np.sum(Device_data_is_within_bounds[train_idx],axis = 0) / len(Device_data[train_idx]) * 100
    Device_data_accuracy_percentage_val = np.sum(Device_data_is_within_bounds[val_idx],axis = 0) / len(Device_data[val_idx]) * 100
    Device_data_accuracy_percentage_test = np.sum(Device_data_is_within_bounds[test_idx],axis = 0) / len(Device_data[test_idx]) * 100
    
    
    # Calculate percentage accuracy
    
    
    PCE_std = propagated_pce_std(Device_data_predict,Device_data_predict_std)
    PCE_predict = np.prod(Device_data_predict,axis=1)/100
    PCE = np.prod(Device_data,axis=1)/100
    # Calculate bounds
    PCE_lower_bound = PCE_predict - 1.96 * PCE_std
    PCE_upper_bound = PCE_predict + 1.96 * PCE_std

    # Check if PCE falls within bounds
    PCE_is_within_bounds = (PCE >= PCE_lower_bound) & (PCE <= PCE_upper_bound)

    # Calculate percentage accuracy
    PCE_accuracy_percentage_train = np.sum(PCE_is_within_bounds[train_idx]) / len(PCE[train_idx]) * 100
    PCE_accuracy_percentage_val = np.sum(PCE_is_within_bounds[val_idx]) / len(PCE[val_idx]) * 100
    PCE_accuracy_percentage_test = np.sum(PCE_is_within_bounds[test_idx]) / len(PCE[test_idx]) * 100
    
    Predicted_accuracy_train = np.append(Device_data_accuracy_percentage_train, PCE_accuracy_percentage_train)
    Predicted_accuracy_val = np.append(Device_data_accuracy_percentage_val, PCE_accuracy_percentage_val)
    Predicted_accuracy_test = np.append(Device_data_accuracy_percentage_test, PCE_accuracy_percentage_test)
    Predicted_accuracy = np.column_stack((Predicted_accuracy_train, Predicted_accuracy_val, Predicted_accuracy_test))
    return Predicted_accuracy
