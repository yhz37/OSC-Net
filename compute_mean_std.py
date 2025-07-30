# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:43:41 2024

@author: haizhouy
"""
import numpy as np
def propagated_pce_std(Device_data_predict,Device_data_predict_std):
    # Extract x, y, z, and their standard deviations
    x = Device_data_predict[:, 0]
    y = Device_data_predict[:, 1]
    z = Device_data_predict[:, 2]
    
    std_x = Device_data_predict_std[:, 0]
    std_y = Device_data_predict_std[:, 1]
    std_z = Device_data_predict_std[:, 2]

    # Calculate propagated variance and standard deviation
    var_pce = (y * z)**2 * std_x**2 + (x * z)**2 * std_y**2 + (x * y)**2 * std_z**2
    std_pce = np.sqrt(var_pce)/100
    return std_pce



def compute_mean_std(Device_data_predict,Device_data_predict_std,train_idx,val_idx,test_idx,Device_data_predict_std_a = None,Device_data_predict_std_e = None):


    PCE_std = propagated_pce_std(Device_data_predict,Device_data_predict_std)
    std_all = np.column_stack((Device_data_predict_std, PCE_std))
    std_all_train = np.mean(std_all[train_idx],axis=0)
    std_all_val = np.mean(std_all[val_idx],axis=0)
    std_all_test = np.mean(std_all[test_idx],axis=0)
    std_all_summary = np.column_stack((std_all_train, std_all_val, std_all_test))
    
    if Device_data_predict_std_a is not None:
        PCE_std_a = propagated_pce_std(Device_data_predict,Device_data_predict_std_a)
        std_all_a = np.column_stack((Device_data_predict_std_a, PCE_std_a))
        std_all_a_train = np.mean(std_all_a[train_idx],axis=0)
        std_all_a_val = np.mean(std_all_a[val_idx],axis=0)
        std_all_a_test = np.mean(std_all_a[test_idx],axis=0)
        std_all_a_summary = np.column_stack((std_all_a_train, std_all_a_val, std_all_a_test))
        
    if Device_data_predict_std_e is not None:
        PCE_std_e = propagated_pce_std(Device_data_predict,Device_data_predict_std_e)
        std_all_e = np.column_stack((Device_data_predict_std_e, PCE_std_e))
        std_all_e_train = np.mean(std_all_e[train_idx],axis=0)
        std_all_e_val = np.mean(std_all_e[val_idx],axis=0)
        std_all_e_test = np.mean(std_all_e[test_idx],axis=0)
        std_all_e_summary = np.column_stack((std_all_e_train, std_all_e_val, std_all_e_test))


    if Device_data_predict_std_a is not None and Device_data_predict_std_e is not None:
        return std_all_summary, std_all_a_summary, std_all_e_summary
    else:
        return std_all_summary
    
        



