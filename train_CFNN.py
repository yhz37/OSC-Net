# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:21:51 2023

@author: haizhouy
"""

from torch.utils.data import DataLoader
from torch import optim
import torch
from tqdm import tqdm
import copy
import numpy as np
import torch.nn as nn

from evaluate import evaluate
from CFNN import CFNN,CFNN_hete
# def initialize_weights(model):
#     for layer in model.modules():
#         if isinstance(layer, nn.Linear):
#             nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#             if layer.bias is not None:
#                 nn.init.constant_(layer.bias, 0)
#         elif isinstance(layer, nn.Conv2d):
#             nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)
#         elif isinstance(layer, nn.Conv1d):
#             nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#             if layer.bias is not None:
#                 nn.init.constant_(layer.bias, 0)
#         elif isinstance(layer, nn.BatchNorm1d):
#             # Optional, if your model adds batch normalization in the future
#             nn.init.constant_(layer.weight, 1)
#             nn.init.constant_(layer.bias, 0)
            
def initialize_weights(module):
    """
    Custom weight initialization function for the CFNN and CFNN_hete models.
    Applies Kaiming Normal initialization to Conv1d and Linear layers.
    """
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            



def initial_net(n_ch,n_cl,device,Num_F = 4096,case ='n_n'):
    if 'Aleatoric_He' in case:
        net = CFNN_hete(n_channels=n_ch, n_classes=n_cl*2, bilinear=False,Num_F=Num_F)
    else:
        net = CFNN(n_channels=n_ch, n_classes=n_cl, bilinear=False,Num_F=Num_F)
    net.apply(initialize_weights)

    # initialize_weights(net)
    net.to(device=device)

    return net


def train_CFNN(train_set,val_set,epochs,batch_size,lr,device,nBits = 4096,case='n_n',net = None, net_idx = 0, total_net = 1):
    if 'Aleatoric_He' in case:
        criterion = nn.GaussianNLLLoss()
    else:
        criterion = nn.MSELoss()

    OSC_Net,Final_lr,scheduler,lc = train_net(device=device,
                                           train_set=train_set,
                                           val_set=val_set,
                                           criterion=criterion,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           learning_rate=lr,
                                           amp=False,
                                           nBits = nBits,
                                           net = net,
                                           case =case,
                                           net_idx = net_idx, 
                                           total_net = total_net)
       
    return OSC_Net,Final_lr,scheduler,lc

def train_net(device,
              train_set,
              val_set,
              criterion,
              epochs: int = 5,
              batch_size: int = 10,
              learning_rate: float = 1e-5,
              amp: bool = False,
              nBits = 4096,
              net = None,
              case ='n_n',
              net_idx = 0, 
              total_net = 1):
    if net==None:
        net = initial_net(train_set.x_data.shape[1],train_set.y_data.shape[1],device,Num_F=nBits,case =case)
    else:
        net.to(device=device)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor =0.993, min_lr=1e-6 )  # work well, back up for batch _3:patience=13, factor =0.999, min_lr=5e-7
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # global_step = 0
    n_train = len(train_set)

    lc = np.empty([epochs,2])
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Net {net_idx+1}/{total_net} - Epoch {epoch + 1}/{epochs}', unit='data') as pbar:
            for batch, (Input, Output) in enumerate(train_loader):
                assert Input.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded Input have {Input.shape[1]} channels. Please check that ' \
                    'the Input are loaded correctly.'

                Input = Input.to(device=device, dtype=torch.float32)
                Output = Output.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    if 'Aleatoric_He' in case:
                        Output_pred,s = net(Input)
                        loss = criterion(Output_pred, Output,s.exp()) 
                    else:
                        Output_pred = net(Input)
                        loss = criterion(Output_pred, Output)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(Input.shape[0])
                # global_step += 1
                epoch_loss += loss.item()*Input.shape[0]

                pbar.set_postfix(**{'loss (Epoch)': epoch_loss})
                del loss
                del Output_pred
                # # Evaluation round
                # division_step = (n_train // (10 * batch_size))
                # if division_step == 0:
                #     division_step = 1
                # if division_step > 0:
                #     if global_step % division_step == 0:
                #         val_score = evaluate(net, val_loader, device,criterion,case)
                #         scheduler.step(val_score)



        lc[epoch,0] = epoch_loss/n_train
        lc[epoch,1] = evaluate(net, val_loader, device,criterion,case)
        scheduler.step(lc[epoch,1])
        print(f"LR: {optimizer.param_groups[0]['lr']}")
        print('train loss = {}'.format(lc[epoch,0]))
        print('val loss = {}'.format(lc[epoch,1]))
    for param_group in optimizer.param_groups:
        Final_lr = param_group['lr']
    return net, Final_lr,scheduler,lc