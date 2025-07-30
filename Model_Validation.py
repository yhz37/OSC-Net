# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:38:19 2025

@author: haizhouy
"""


import sys
sys.path.insert(0, 'D:\\OneDrive - Michigan Medicine\\Python\\Organic_Solar_Cell\\Database')
sys.path.insert(0, 'D:\\OneDrive - Michigan Medicine\\Python\\CarVas')

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import scipy.io as sio
from scipy import stats
from predictor_MF_Unet_NB import predictor_MF_Unet_NB
from train_CFNN import train_CFNN
from get_args_MF_OSC import get_args_MF_OSC
from Dataset import Dataset
from evaluate import evaluate
from normalize_data import normalize_data
from denormalize_data_mm import denormalize_data_mm
from normalize_data_mm import normalize_data_mm
from OSC_Plot import OSC_Plot
from loss_curve_plot import loss_curve_plot
from mse_r_compute import mse_r_compute
from compute_mean_std import compute_mean_std
from compute_conf import compute_conf
from ConfInteAccu import ConfInteAccu


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(r'D:\OneDrive - Michigan Medicine\Python\Organic_Solar_Cell\Results\MF_OSC_2048_s7_Aleatoric_He_MAP_nets5_p500_f7000_64_128_32_8_512_128.pkl','rb') as f:  # Python 3: open(..., 'rb')
    CFNN, Device_data_predict, Device_data, train_idx, val_idx, test_idx, \
    max_ratio, min_ratio, max_Device_data, min_Device_data, Final_lr, lc, \
    input, device, mse_summary, r_summary, CFNN_pre, \
    Device_data_computational_predict, Device_data_computational, \
    train_idx_computational, val_idx_computational, test_idx_computational, \
    Final_lr_pre, lc_pre, input_computational, \
    mse_summary_compu_pre, r_summary_compu_pre, \
    mse_summary_pre, r_summary_pre,case = pickle.load(f)

loss_curve_plot(lc_pre, label = 'pretrain')
loss_curve_plot(lc, label = 'finetune')
args = get_args_MF_OSC()
args.case = case

# Find all files that match the pattern
matching_files = r'D:\OneDrive - Michigan Medicine\Python\Organic_Solar_Cell\Database\OSC_data_bits2048_ir512_374.mat'


OSC_data = sio.loadmat(matching_files)
ratio = OSC_data['ratio']
Device_data = OSC_data['Device_data']
donor_binary = OSC_data['donor_binary']
acceptor_binary = OSC_data['acceptor_binary']
row_index = OSC_data['row_index']

Device_data = Device_data[:,[0,2,4]]  #VOC(V),JSC(mA/cm2),FF(%)
scaled_ratio = normalize_data_mm(ratio,max_ratio,min_ratio)
scaled_Device_data = normalize_data_mm(Device_data,max_Device_data,min_Device_data)

scaled_ratio = scaled_ratio.reshape(scaled_ratio.shape[0],1,-1)
donor_binary = donor_binary.reshape(donor_binary.shape[0],1,-1)
acceptor_binary = acceptor_binary.reshape(acceptor_binary.shape[0],1,-1)
scaled_Device_data = scaled_Device_data.reshape(scaled_Device_data.shape[0],-1)

ratio_input = np.concatenate((scaled_ratio, scaled_ratio), axis=1)
smile_input = np.concatenate((donor_binary, acceptor_binary), axis=1)
input = np.concatenate((smile_input,ratio_input), axis=2)

if '_MAP' in args.case:
    if 'Aleatoric_He' in args.case: 
        scaled_Device_data_predict,scaled_Device_data_predict_std_a , scaled_Device_data_predict_std_e, scaled_Device_data_predict_std = predictor_MF_Unet_NB(torch.from_numpy(input), CFNN, args.case, device)
    else:
        scaled_Device_data_predict,scaled_Device_data_predict_std = predictor_MF_Unet_NB(torch.from_numpy(input), CFNN, args.case, device)

else:
    scaled_Device_data_predict = predictor_MF_Unet_NB(torch.from_numpy(input), CFNN, args.case, device)

scaled_Device_data_predict = scaled_Device_data_predict.cpu().detach().numpy()
Device_data_predict = denormalize_data_mm(scaled_Device_data_predict,max_Device_data,min_Device_data)
test_idx = np.array([-4, -3, -2, -1], dtype=int)
if '_MAP' in args.case:
    if 'Aleatoric_He' in args.case: 
        
        scaled_Device_data_predict_std_a = scaled_Device_data_predict_std_a.cpu().detach().numpy()
        scaled_Device_data_predict_std_e = scaled_Device_data_predict_std_e.cpu().detach().numpy()
        scaled_Device_data_predict_std = scaled_Device_data_predict_std.cpu().detach().numpy()
        
        Device_data_predict_std_a = scaled_Device_data_predict_std_a*(max_Device_data-min_Device_data)
        Device_data_predict_std_e = scaled_Device_data_predict_std_e*(max_Device_data-min_Device_data)
        Device_data_predict_std = scaled_Device_data_predict_std*(max_Device_data-min_Device_data)
        OSC_Plot(Device_data,Device_data_predict,train_idx,val_idx,test_idx,min_Device_data,max_Device_data,model_type='finetune',Device_data_predict_std=Device_data_predict_std,Device_data_predict_std_a=Device_data_predict_std_a,data_source='Experimental')
    else:
        scaled_Device_data_predict_std = scaled_Device_data_predict_std.cpu().detach().numpy()
        Device_data_predict_std = scaled_Device_data_predict_std*(max_Device_data-min_Device_data)
        OSC_Plot(Device_data,Device_data_predict,train_idx,val_idx,test_idx,min_Device_data,max_Device_data,model_type='finetune',Device_data_predict_std=Device_data_predict_std,data_source='Experimental')

else:
   OSC_Plot(Device_data,Device_data_predict,train_idx,val_idx,test_idx,min_Device_data,max_Device_data,model_type='finetune',data_source='Experimental')

mse_summary,r_summary = mse_r_compute(Device_data,Device_data_predict,train_idx,val_idx,test_idx)


if '_MAP' in args.case:
    if 'Aleatoric_He' in args.case: 
        std_all_summary, std_all_a_summary, std_all_e_summary = compute_mean_std(Device_data_predict,Device_data_predict_std,train_idx,val_idx,test_idx,Device_data_predict_std_a = Device_data_predict_std_a,Device_data_predict_std_e = Device_data_predict_std_e)
    else:
        std_all_summary = compute_mean_std(Device_data_predict,Device_data_predict_std,train_idx,val_idx,test_idx)

        

conf_matrix_train, conf_matrix_val, conf_matrix_test = compute_conf(Device_data,Device_data_predict,train_idx,val_idx,test_idx)

CI_accuracy = ConfInteAccu(Device_data, Device_data_predict,Device_data_predict_std,train_idx,val_idx,test_idx)

