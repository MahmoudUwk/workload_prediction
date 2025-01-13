import os
import sys
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Alibaba_helper_functions import loadDatasetObj,RMSE,save_object,flatten,MAE,MAPE
import pandas as pd
from sklearn.metrics import r2_score


def write_txt(txt,fname):
    f = open(fname, "w")
    f.write(txt)
    f.close()

from args_google import get_paths
base_path,_,_,_,_,working_path,sav_path = get_paths()
# sav_path =  os.path.join(working_path,'plots')

if not os.path.exists(sav_path):
    os.makedirs(sav_path)

result_files = os.listdir(working_path)


algorithms = ['CDEL_google_inference','Adaptive_predictor','HistGradientBoostingRegressor','SVR','LinearRegression']

result_files = [file for file in result_files if file.endswith(".obj")]

result_files = [file for c1,alg in enumerate(algorithms) for c2,file in enumerate(result_files) if alg in file]



alg_rename = {'CDEL_google_inference':'CEDL (Proposed)',
              'LinearRegression':'LR',
              'Adaptive_predictor':'Adaptive selector',
              'HistGradientBoostingRegressor':'GBT',
              'SVR':'SVR',}

full_file_path = [os.path.join(working_path,file) for file in result_files]

indeces = [algorithms[c2] for c1,file in enumerate(result_files) for c2,alg in enumerate(algorithms) if alg in file]



# full_file_path = [file for file in full_file_path if os.path.exists(file)]

# result_files = [file.split('.')[0] for file in result_files]
# indeces_short = [alg_rename_short[f_i] for f_i in result_files]
indeces = [alg_rename[f_i] for f_i in indeces]
indeces_short =indeces
colors = [ '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4','#e6194b']




fs = 17
#%%
#scatter plots
fig, axs = plt.subplots(2,int(len(result_files)/2), figsize=(20,8),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .3, wspace=.08)

axs = axs.ravel()
ind_last_scatter = int(len(result_files)/2)*2

for i,res_file in enumerate(full_file_path[:ind_last_scatter]):
    results_i = loadDatasetObj(res_file)
    a1 = np.clip(flatten(results_i['y_test']),0,100)
    a2 = np.clip(flatten(results_i['y_test_pred']),0,100)
    axs[i].scatter(a1,a2 ,alpha=0.3,s=15, color = colors[i],marker='o', linewidth=1.5)
    axs[i].set_title(indeces[i], fontsize=fs+4, x=0.35, y=0.75)

    max_val_i = max(np.max(a1),np.max(a2))
    X_line = np.arange(0,max_val_i,max_val_i/200)
    axs[i].plot(X_line,X_line)
    axs[i].tick_params(axis='x', labelsize=fs)
    axs[i].tick_params(axis='y', labelsize=fs)
    plt.grid()
    axs[i].set_xlabel('Real CPU Utilization(%)', fontsize=fs-1)

    if i == 2  or i == 0:
        axs[i].set_ylabel('Predicted CPU (%)', fontsize=fs)
    
plt.savefig(os.path.join(sav_path,'scatter_plot.png'),bbox_inches='tight')

#%% bar plot RMSE
# patterns = [ "||" ,  "/",  "x","-","+",'//']
barWidth = 0.3
fig = plt.figure(figsize=(14,8),dpi=150)


data_res = []
train_time_all = []
test_time_all = []
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    a1 = flatten(results_i['y_test'])
    a2 = np.clip(flatten(results_i['y_test_pred']),0,100)
    # train_time_all.append(results_i['train_time'])
    # test_time_all.append(results_i['test_time'])
    print(np.max(a1),np.min(a1),np.max(a2),np.min(a2))
    RMSE_i = RMSE(a1,a2)
    MAE_i = MAE(a1,a2)
    MAPE_i = MAPE(a1,a2)
    r2_sc = r2_score(a1,a2)
    data_res.append([RMSE_i,MAE_i,MAPE_i,r2_sc])


    plt.bar(counter , RMSE_i , color=colors[counter],
            width=barWidth, hatch="x", edgecolor='black', label = indeces[counter])

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontsize=fs)
plt.ylabel('RMSE', fontsize=fs)
# plt.yticks(labelsize=fs)
# plt.xticks(labelsize=fs)

plt.xticks([r for r in range(len(indeces_short))], indeces_short, fontsize=fs)
_,b = plt.ylim()
plt.ylim([0,b*1.1])
# yticks_no = np.arange(0,lim,steps)
# yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
# plt.yticks(yticks_no,yticks)
# plt.title('Bar plot of the RMSE for different algorithms',fontsize=fs+2)
# Create legend & Show graphic
plt.legend(prop={'size': fs},loc=(0.05,0.5))
plt.gca().grid(True)
plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
plt.savefig(os.path.join(sav_path,'bar_plot.eps'),bbox_inches='tight', format='eps')

#%% train and test times
Metric = ['RMSE','MAE',"MAPE","R2_score"]
data_res = np.array(data_res)

def process_time(train_time_all):
    train_time_all2 = []
    for TT in train_time_all:
        if isinstance(TT, list):
            TT = sum(TT)
        if TT < 1 :
            TT = TT*60
            TT = str(np.round(TT,1))+' s'
        else:
            if TT < 60:
                TT = str(np.round(TT,1))+' min'
            else:
                TT = TT / 60
                TT = str(np.round(TT,1))+' hr'
        train_time_all2.append(TT)
    return train_time_all2

data_res = np.round(np.array(data_res),2)
# TT = np.array([process_time(train_time_all),process_time(test_time_all)]).T
# dat = np.concatenate((data_res, TT),axis=1)
df = pd.DataFrame(data=data_res,columns=Metric,index = indeces)
print(df)
latex_txt = df.style.to_latex()


write_txt(latex_txt,os.path.join(sav_path,'results_table_latex.txt'))

#%%
def get_ind_plot(y,i):
    mean_test = np.argsort([np.mean(m_y) for m_y in y])
    var_test = np.argsort([np.var(m_y) for m_y in y])
    high_load_ind = mean_test[-1]
    low_load_ind = mean_test[2]
    high_var_ind = var_test[-1]
    low_var_ind = var_test[1]
    ind_plotMs = [high_load_ind,low_load_ind,high_var_ind,low_var_ind]
    return ind_plotMs[i]

def append_arr(y_true,y_pred):
    return np.concatenate([y_true[:(len(y_true)-len(y_pred))],y_pred],axis=0)

length_plot = 150
titles_plot = ['High Load','Low Load', 'High Variation', 'Low Variation']
fig, axs = plt.subplots(len(titles_plot),1, figsize=(20,11),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .3, wspace=.001)
axs = axs.ravel()
alpha_blue = 0.35
markers = ['x','','','','']
y_cp = results_i['y_test'].copy()
for i,title_i in enumerate(titles_plot):
    for j,res_file in enumerate(full_file_path):
        results_i = loadDatasetObj(res_file)
        ind_machine_id = get_ind_plot(y_cp,i)
        print(indeces[j],ind_machine_id)
        y_pred_plot = append_arr(y_cp[ind_machine_id],
                                 np.squeeze(results_i['y_test_pred'][ind_machine_id]))[:length_plot]
        y_pred_plot = np.clip(y_pred_plot,0,100)
        if i == 3:
            axs[i].plot(y_pred_plot, markers[j]+'-', 
                        color = colors[j],alpha=0.6, linewidth=0.8,label=indeces[j])
        else:
            axs[i].plot(y_pred_plot, markers[j]+'-', 
                        color = colors[j],alpha=0.6, linewidth=0.8)
    if i == 3:
        axs[i].plot(np.squeeze(y_cp[ind_machine_id][:length_plot]), 'o-', 
        color = 'blue',markersize=5, linewidth=0.8, alpha = alpha_blue,label='true')
    else:
        axs[i].plot(np.squeeze(y_cp[ind_machine_id][:length_plot]), 'o-', 
            color = 'blue',markersize=5, linewidth=0.8, alpha = alpha_blue)
    len_i = len(np.squeeze(y_cp[ind_machine_id])[:length_plot])
    axs[i].set_title(titles_plot[i], fontsize=fs-2, x=0.6, y=0.45)
    axs[i].set_xticks(np.arange(0,len_i,10) , 5*np.arange(0,len_i,10) )

    
    if i == 2:
        axs[i].set_ylabel('CPU utilization (%)', fontsize=fs+2)

    

axs[i].set_xlabel('Timestamp (min)' ,fontsize=fs)

fig.legend(prop={'size': fs-1},loc=(0.75,0.61))

M_id = np.array(results_i['Mids_test'])[ind_machine_id]
# fig.suptitle('Prediction vs true values for machine number '+str(M_id), fontsize=fs, x=0.5, y=0.92)

plt.savefig(os.path.join(sav_path,'PredictionsVsReal.eps'),bbox_inches='tight', format='eps')


