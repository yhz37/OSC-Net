# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:08:18 2024

@author: haizhouy
"""
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)
plt.ion()

# Function to blend two colors
def blend_color(color1, color2, blend_factor=0.5):
    blended_color = [
        blend_factor * c1 + (1 - blend_factor) * c2
        for c1, c2 in zip(color1, color2)
    ]
    return blended_color

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

def OSC_Plot(Device_data,Device_data_predict,train_idx,val_idx,test_idx,min_Device_data,max_Device_data,model_type='pre',Device_data_predict_std=None,Device_data_predict_std_a = None,data_source='Experimental'):


    labels = [r"$V_{\mathrm{OC}}(V)$", r"$J_{\mathrm{SC}}(\mathrm{mA} \cdot \mathrm{cm}^{-2})$", r"$FF(\%)$", r"$\mathrm{PCE}(\%)$"]
    filenames = ["VOC", "JSC", "FF", "PCE"]
    colors = ['green','red','blue','purple']
    ecolors = ['lightgreen','pink','lightblue','violet']
    plt.ioff()
    for idx in range (Device_data.shape[1]):
        fig = plt.figure()

        fig.set_size_inches(6, 5)
        plt.plot([min_Device_data[idx],max_Device_data[idx]],[min_Device_data[idx],max_Device_data[idx]],c ='k',linewidth = 0.3)
       
        if Device_data_predict_std is None:
            # plt.scatter(Device_data[train_idx,idx],Device_data_predict[train_idx,idx],label = 'train',s=3,c = 'blue')
            # plt.scatter(Device_data[val_idx,idx],Device_data_predict[val_idx,idx],label = 'val',s=3,c = 'red')
            plt.scatter(Device_data[test_idx,idx],Device_data_predict[test_idx,idx],label = 'test',s=3, c=colors[idx])
        else:
                
            # plt.errorbar(Device_data[train_idx, idx], Device_data_predict[train_idx, idx], 
            #  yerr=Device_data_predict_std[train_idx, idx]*1.96,
            #  fmt='o', markersize=5, color='blue', ecolor='lightblue', 
            #  elinewidth=1, capsize=3,label = 'train_UQ')
            
            # if Device_data_predict_std_a is not None:
            #     # Convert colors to normalized RGB
            #     color_blue = np.array([0, 0, 1])  # normalized values of (0, 0, 255)
            #     color_lightblue = np.array([173/255, 216/255, 230/255])
            #     # Blend colors
            #     blended_color_blue = blend_color(color_blue, color_lightblue, blend_factor=0.5)                
            #     plt.errorbar(Device_data[train_idx, idx], Device_data_predict[train_idx, idx], 
            #      yerr=Device_data_predict_std_a[train_idx, idx]*1.96,
            #      fmt='o', markersize=5, color='blue', ecolor=blended_color_blue, 
            #      elinewidth=1, capsize=3,label = 'train_UQa')
                    
            # plt.errorbar(Device_data[val_idx, idx], Device_data_predict[val_idx, idx], 
            #  yerr=Device_data_predict_std[val_idx, idx]*1.96,
            #  fmt='o', markersize=5, color='red', ecolor='pink', 
            #  elinewidth=1, capsize=3,label = 'val_UQ')
            
            # if Device_data_predict_std_a is not None:
            #     color_red = np.array([255/255, 0/255, 0/255])  # normalized values of (255, 0, 0)
            #     color_pink = np.array([255/255, 192/255, 203/255])
            #     # Blend colors
            #     blended_color_red = blend_color(color_red, color_pink, blend_factor=0.5)  
            #     plt.errorbar(Device_data[val_idx, idx], Device_data_predict[val_idx, idx], 
            #      yerr=Device_data_predict_std_a[val_idx, idx]*1.96,
            #      fmt='o', markersize=5, color='red', ecolor=blended_color_red, 
            #      elinewidth=1, capsize=3,label = 'val_UQa')   
                         
                
            plt.errorbar(Device_data[test_idx, idx], Device_data_predict[test_idx, idx], 
             yerr=Device_data_predict_std[test_idx, idx]*1.96,
             fmt='o', markersize=5, color=colors[idx], ecolor=ecolors[idx], 
             elinewidth=1, capsize=3,label = 'Total UQ')
            
            if Device_data_predict_std_a is not None:
                # color_green = np.array([0/255, 128/255, 0/255])  # normalized values of (0, 128, 0)
                # color_lightgreen = np.array([144/255, 238/255, 144/255])
                # Blend colors
                # blended_color_green = blend_color(color_green, color_lightgreen, blend_factor=0.5)  
                plt.errorbar(Device_data[test_idx, idx], Device_data_predict[test_idx, idx], 
                 yerr=Device_data_predict_std_a[test_idx, idx]*1.96,
                 fmt='o', markersize=5, color=colors[idx], ecolor=colors[idx], 
                 elinewidth=1, capsize=3,label = 'Aleatoric UQ')
        
        
        plt.xticks(fontsize=18,weight='bold')
        plt.yticks(fontsize=18,weight='bold')
        plt.xlabel("{} {}".format(data_source,labels[idx]),fontsize=24,weight='bold')
        plt.ylabel("Predicted {}".format(labels[idx]),fontsize=24,weight='bold')
        plt.legend(shadow=True,prop={'weight':'bold','size':15})
        plt.ticklabel_format(useOffset=False)
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.15)
        plt.locator_params(nbins=5)
        resolution_value = 1200
        plt.savefig("{}_{}_{}.jpg".format(filenames[idx],model_type,data_source), format="jpg", dpi=resolution_value)
        # plt.show()
        plt.close(fig)
        
        
    if Device_data_predict_std is not None:
        
        std_pce =propagated_pce_std(Device_data_predict,Device_data_predict_std)
        if Device_data_predict_std_a is not None:
            std_a_pce = propagated_pce_std(Device_data_predict,Device_data_predict_std_a)

    fig = plt.figure()
    plt.ioff()
    fig.set_size_inches(6, 5)
    plt.plot([np.min(np.prod(Device_data,axis=1)/100),np.max(np.prod(Device_data,axis=1))/100],[np.min(np.prod(Device_data,axis=1)/100),np.max(np.prod(Device_data,axis=1)/100)],c ='k',linewidth = 0.3)

    if Device_data_predict_std is None:
        # plt.scatter(np.prod(Device_data[train_idx],axis=1)/100,np.prod(Device_data_predict[train_idx],axis=1)/100,label = 'train',s=3,c = 'blue')
        # plt.scatter(np.prod(Device_data[val_idx],axis=1)/100,np.prod(Device_data_predict[val_idx],axis=1)/100,label = 'val',s=3,c = 'red')
        plt.scatter(np.prod(Device_data[test_idx],axis=1)/100,np.prod(Device_data_predict[test_idx],axis=1)/100,label = 'test',s=3, c='green')
    else:
        # plt.errorbar(np.prod(Device_data[train_idx],axis=1)/100, np.prod(Device_data_predict[train_idx],axis=1)/100, 
        #  yerr=std_pce[train_idx]*1.96,
        #  fmt='o', markersize=5, color='blue', ecolor='lightblue', 
        #  elinewidth=1, capsize=3,label = 'train_UQ')
        # if Device_data_predict_std_a is not None:
        #     plt.errorbar(np.prod(Device_data[train_idx],axis=1)/100, np.prod(Device_data_predict[train_idx],axis=1)/100, 
        #      yerr=std_a_pce[train_idx]*1.96,
        #      fmt='o', markersize=5, color='blue', ecolor=blended_color_blue, 
        #      elinewidth=1, capsize=3,label = 'train_UQa')
            
        
        # plt.errorbar(np.prod(Device_data[val_idx],axis=1)/100, np.prod(Device_data_predict[val_idx],axis=1)/100, 
        #  yerr=std_pce[val_idx]*1.96,
        #  fmt='o', markersize=5, color='red', ecolor='pink', 
        #  elinewidth=1, capsize=3,label = 'val_UQ')
        # if Device_data_predict_std_a is not None:
        #     plt.errorbar(np.prod(Device_data[val_idx],axis=1)/100, np.prod(Device_data_predict[val_idx],axis=1)/100, 
        #      yerr=std_a_pce[val_idx]*1.96,
        #      fmt='o', markersize=5, color='red', ecolor=blended_color_red, 
        #      elinewidth=1, capsize=3,label = 'val_UQa')
        
        
        plt.errorbar(np.prod(Device_data[test_idx],axis=1)/100, np.prod(Device_data_predict[test_idx],axis=1)/100, 
         yerr=std_pce[test_idx]*1.96,
         fmt='o', markersize=5, color=colors[-1], ecolor=ecolors[-1], 
         elinewidth=1, capsize=3,label = 'Total UQ')
        if Device_data_predict_std_a is not None:
            plt.errorbar(np.prod(Device_data[test_idx],axis=1)/100, np.prod(Device_data_predict[test_idx],axis=1)/100, 
             yerr=std_a_pce[test_idx]*1.96,
             fmt='o', markersize=5, color=colors[-1], ecolor=colors[-1], 
             elinewidth=1, capsize=3,label = 'Aleatoric UQ')
    
    plt.xticks(fontsize=18,weight='bold')
    plt.yticks(fontsize=18,weight='bold')
    plt.xlabel("{} {}".format(data_source,labels[-1]),fontsize=24,weight='bold')
    plt.ylabel("Predicted {}".format(labels[-1]),fontsize=24,weight='bold')
    plt.legend(shadow=True,prop={'weight':'bold','size':15})
    plt.ticklabel_format(useOffset=False)
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(left=0.15)
    plt.locator_params(nbins=5)
    resolution_value = 1200
    plt.savefig("{}_{}_{}.jpg".format(filenames[-1],model_type,data_source), format="jpg", dpi=resolution_value)
    # plt.show()
    plt.close(fig)