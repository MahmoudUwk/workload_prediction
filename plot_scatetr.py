import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle 
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def flatten(xss):
    return np.array([x for xs in xss for x in xs])

def shift_right(lst, n):
    n = n % len(lst)
    return lst[-n:] + lst[:-n]
dataset_plot = 0

colors = sns.color_palette("Set1", 10).as_hex()
colors = shift_right(colors, -1)
colors[4], colors[-1] = colors[-1], colors[4]
colors[4], colors[5] = colors[5], colors[4]



data_sets = ['Alibaba','BB']
dataset_names_label = ['Alibaba','Bitbrains']


if data_sets[dataset_plot] == 'BB':
    from args_BB import get_paths
    _,_,_,_,_,working_path,sav_path = get_paths()
    CEDL_name = 'BB'
    x_legend,y_legend = 0.75,0.65
    x_title,y_title = 0.11,0.74
    table_label = 'table:results_BB'
    ipTS = [-1, 0, -2 ,0]
    spInd = 1000
elif data_sets[dataset_plot] == 'Alibaba':
    from args import get_paths
    _,_,_,_,_,working_path,sav_path = get_paths()
    CEDL_name = 'Alibaba'#'cuckoosearch_Ali'
    x_legend,y_legend = 0.81,0.3
    x_title,y_title = 0.45,0.72
    table_label = 'table:res'
    ipTS = [-1, 1, -1 ,2]
    spInd = 32
    

if not os.path.exists(sav_path):
    os.makedirs(sav_path)

result_files = os.listdir(working_path)



algorithms = [CEDL_name+'TST_LSTM',CEDL_name+'EnDeAtt',CEDL_name+'LSTM','PatchTST',CEDL_name+'CNN','Adaptive_predictor','HistGradientBoostingRegressor','SVR','LinearRegression']

result_files = [file for file in result_files if file.endswith(".obj")]

result_files = [file for c1,alg in enumerate(algorithms) for c2,file in enumerate(result_files) if alg in file]



alg_rename = {CEDL_name+'TST_LSTM':'TempoSight',
              CEDL_name+'EnDeAtt':'CEDL',
              CEDL_name+'LSTM':'LSTM',
              'PatchTST':'Patch\nTST',
              CEDL_name+'CNN':'CNN',
              'Adaptive_predictor':'Adaptive',
              'HistGradientBoostingRegressor':'GBT',
              'SVR':'SVR',
              'LinearRegression':'LR',}

full_file_path = [os.path.join(working_path,file) for file in result_files]

# indeces = [algorithms[c2] for c1,file in enumerate(result_files) for c2,alg in enumerate(algorithms) if alg in file]



# full_file_path = [file for file in full_file_path if os.path.exists(file)]

# result_files = [file.split('.')[0] for file in result_files]
# indeces_short = [alg_rename_short[f_i] for f_i in result_files]
indeces = list(alg_rename.values()) #[alg_rename[f_i] for f_i in indeces]
indeces_short =indeces


fs = 17


#scatter plots
ind_last_scatter = 4
if data_sets[dataset_plot] == 'Alibaba':
    fig, axs = plt.subplots(2,int(ind_last_scatter/2), figsize=(20,8),dpi=500, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .3, wspace=.08)
    
    axs = axs.ravel()
    #int(len(result_files)/2)*2
    
    for i,res_file in enumerate(full_file_path[:ind_last_scatter]):
        results_i = loadDatasetObj(res_file)
        scal1 = 1
        if np.max(results_i['y_test'][0])<2:
            scal1 = 100
        a1 = scal1*np.clip(flatten(results_i['y_test']),0,100)
        a2 = np.clip(flatten(results_i['y_test_pred']),0,100)
        axs[i].scatter(a1,a2 ,alpha=0.2,s=15, color = colors[i],marker='o'
                       , linewidth=1.5)
        axs[i].set_title(indeces[i], fontsize=fs+4, x=0.35, y=0.75)
    
        max_val_i = max(np.max(a1),np.max(a2))
        X_line = np.arange(0,max_val_i,max_val_i/200)
        axs[i].plot(X_line,X_line)
        axs[i].tick_params(axis='x', labelsize=fs+1)
        axs[i].tick_params(axis='y', labelsize=fs+1)
        plt.grid()
        axs[i].set_xlabel('Real CPU Utilization(%)', fontsize=fs+2)
    
        if i == 2  or i == 0:
            axs[i].set_ylabel('Predicted CPU (%)', fontsize=fs)
        
        axs[i].grid()
    plt.savefig(os.path.join(sav_path,'scatter_plot.png'),bbox_inches='tight')
    
