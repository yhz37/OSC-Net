# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 23:19:08 2024

@author: haizhouy
"""
import numpy as np
from sklearn.metrics import confusion_matrix

def classify_PCE_values(PCE_values):
    # Define three classes based on PCE values
    classes = np.digitize(PCE_values, bins=[0, 5, 10])
    return classes
def compute_conf(Device_data,Device_data_predict,train_idx,val_idx,test_idx):
    # Calculate PCE and PCE_predict
    PCE = np.prod(Device_data, axis=1) / 100
    PCE_predict = np.prod(Device_data_predict, axis=1) / 100

    # Classify PCE and PCE_predict values into three classes
    PCE_classes = classify_PCE_values(PCE)
    PCE_predict_classes = classify_PCE_values(PCE_predict)

    # Compute confusion matrices for train, validation, and test sets
    conf_matrix_train = confusion_matrix(PCE_classes[train_idx], PCE_predict_classes[train_idx], labels=[1, 2, 3])
    conf_matrix_val = confusion_matrix(PCE_classes[val_idx], PCE_predict_classes[val_idx], labels=[1, 2, 3])
    conf_matrix_test = confusion_matrix(PCE_classes[test_idx], PCE_predict_classes[test_idx], labels=[1, 2, 3])

    return conf_matrix_train, conf_matrix_val, conf_matrix_test