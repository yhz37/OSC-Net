# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:35:41 2024

@author: haizhouy
"""

import argparse

def get_args_MF_OSC():
    parser = argparse.ArgumentParser(description='Train the Net on data')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=7000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--epochs_pre', '-e_pre', metavar='E_pre', type=int, default=500, help='Number of epochs for pretrain')
    parser.add_argument('--batch-size_pre', '-b_pre', dest='batch_size_pre', metavar='B', type=int, default=20, help='Batch size')
    parser.add_argument('--learning-rate_pre', '-l_pre', metavar='LR_pre', type=float, default=1e-3,
                        help='Learning rate_pre', dest='lr_pre')
    
    parser.add_argument('--case', type=str, 
                        default= 'Aleatoric_He_MAP'#'NA_MAP'
                        ,help='type of case study')
    parser.add_argument('--seed', '-s', dest='seed', metavar='s', type=int, default=7)
    parser.add_argument('--num_nets', '-nn', dest='num_nets', metavar='nn', type=int, default=5)
    parser.add_argument('--nBits', '-nBits', dest='nBits', metavar='nBits', type=int, default=2048)

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()