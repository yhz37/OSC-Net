# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 21:40:17 2024

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
import glob
from OSC_Plot import OSC_Plot
from loss_curve_plot import loss_curve_plot
from mse_r_compute import mse_r_compute
import copy



plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)
args = get_args_MF_OSC()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#########################################################Experimental Data
file_path_pattern = r'D:\OneDrive - Michigan Medicine\Python\Organic_Solar_Cell\Database\OSC_data_bits{}_*1782.mat'.format(args.nBits)

# Find all files that match the pattern
matching_files = glob.glob(file_path_pattern)


OSC_data = sio.loadmat(matching_files[0])
ratio = OSC_data['ratio']
Device_data = OSC_data['Device_data']
donor_binary = OSC_data['donor_binary']
acceptor_binary = OSC_data['acceptor_binary']
row_index = OSC_data['row_index']

Device_data = Device_data[:,[0,2,4]]  #VOC(V),JSC(mA/cm2),FF(%)
         
scaled_ratio,max_ratio,min_ratio = normalize_data(ratio)
scaled_Device_data,max_Device_data,min_Device_data = normalize_data(Device_data)

scaled_ratio = scaled_ratio.reshape(scaled_ratio.shape[0],1,-1)
donor_binary = donor_binary.reshape(donor_binary.shape[0],1,-1)
acceptor_binary = acceptor_binary.reshape(acceptor_binary.shape[0],1,-1)
scaled_Device_data = scaled_Device_data.reshape(scaled_Device_data.shape[0],-1)

indices = np.random.RandomState(seed=args.seed).permutation(scaled_ratio.shape[0])
train_idx, val_idx, test_idx = indices[:int(0.9*scaled_ratio.shape[0])],indices[int(0.9*scaled_ratio.shape[0]):int(0.95*scaled_ratio.shape[0])], indices[int(0.95*scaled_ratio.shape[0]):]

ratio_input = np.concatenate((scaled_ratio, scaled_ratio), axis=1)
smile_input = np.concatenate((donor_binary, acceptor_binary), axis=1)
input = np.concatenate((smile_input,ratio_input), axis=2)

train_set = Dataset(input[train_idx],scaled_Device_data[train_idx])
val_set = Dataset(input[val_idx],scaled_Device_data[val_idx])
test_set = Dataset(input[test_idx],scaled_Device_data[test_idx])


if '_MAP' in args.case:
    CFNN = [None] * args.num_nets
    lc = [None] * args.num_nets
    Final_lr = np.empty([args.num_nets,1])
    for idx in range(args.num_nets):

        
        CFNN_single,Final_lr_single,scheduler_single,lc_single = train_CFNN(train_set,val_set,args.epochs,args.batch_size,args.lr,device,args.nBits,case=args.case, net_idx = idx, total_net = args.num_nets )
        CFNN[idx] = copy.deepcopy(CFNN_single)
        Final_lr[idx,0] = Final_lr_single
        lc[idx] = lc_single  
else:
    CFNN,Final_lr,scheduler,lc = train_CFNN(train_set,val_set,args.epochs,args.batch_size,args.lr,device,args.nBits,case=args.case )


loss_curve_plot(lc)

if '_MAP' in args.case:
    if 'Aleatoric_He' in args.case: 
        scaled_Device_data_predict,scaled_Device_data_predict_std_a , scaled_Device_data_predict_std_e, scaled_Device_data_predict_std = predictor_MF_Unet_NB(torch.from_numpy(input), CFNN, args.case, device)
    else:
        scaled_Device_data_predict,scaled_Device_data_predict_std = predictor_MF_Unet_NB(torch.from_numpy(input), CFNN, args.case, device)

else:
    scaled_Device_data_predict = predictor_MF_Unet_NB(torch.from_numpy(input), CFNN, args.case, device)

scaled_Device_data_predict = scaled_Device_data_predict.cpu().detach().numpy()
Device_data_predict = denormalize_data_mm(scaled_Device_data_predict,max_Device_data,min_Device_data)

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

with open('SF_OSC_{}_s{}_{}_nets{}_ep{}_h{}.pkl'.format(args.nBits,args.seed,args.case,args.num_nets,args.epochs,Device_data.shape[0],), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([CFNN,Device_data_predict,Device_data,train_idx, val_idx, test_idx,
                 max_ratio,min_ratio,max_Device_data,min_Device_data, Final_lr,
                 lc,input,device,mse_summary,r_summary,args.case], f)    

