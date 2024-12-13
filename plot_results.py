
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Alibaba_helper_functions import loadDatasetObj,RMSE,save_object,flatten,MAE,MAPE
import pandas as pd

# voltage, 460
# current, 52.5
def write_txt(txt,fname):
    f = open(fname, "w")
    f.write(txt)
    f.close()

base = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba'
working_path = os.path.join(base,'pred_results_all')
sav_path = working_path+'/plots'

if not os.path.exists(sav_path):
    os.makedirs(sav_path)

result_files = os.listdir(working_path)

algorithms = ['Firefly','CuckooSearch','LinearRegression','base_proposed','HistGradientBoostingRegressor','SVR']

result_files = [file for file in result_files if file.endswith(".obj")]

result_files = [file for c1,alg in enumerate(algorithms) for c2,file in enumerate(result_files) if alg in file]



alg_rename = {'CuckooSearch':'En De LSTM\n Cuckoo',
              'LinearRegression':'LR',
              'Firefly':'En De LSTM\n FF',
              'base_proposed':'Adaptive selector',
              'HistGradientBoostingRegressor':'GBT',
              'SVR':'SVR',
              'MonkeyKingEvolutionV3':'LSTM MKE_V3'}

full_file_path = [os.path.join(working_path,file) for file in result_files]

indeces = [algorithms[c2] for c1,file in enumerate(result_files) for c2,alg in enumerate(algorithms) if alg in file]



# full_file_path = [file for file in full_file_path if os.path.exists(file)]

# result_files = [file.split('.')[0] for file in result_files]
# indeces_short = [alg_rename_short[f_i] for f_i in result_files]
indeces = [alg_rename[f_i] for f_i in indeces]
indeces_short =indeces
colors = ['#047495','#632de9','#a4be5c','#acfffc','#ac9362','#3c4d03']

fs = 17
#%%
#scatter plots
fig, axs = plt.subplots(2,int(len(result_files)/2), figsize=(20,8),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .2, wspace=.001)

axs = axs.ravel()

for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    # if 'CuckooSearch' in res_file:
    #     ind_machine_id_cockoo = np.argmin(results_i['rmse_list'])
    #     M_id_plot = results_i['Mids_test'][ind_machine_id_cockoo]
    a1 = flatten(results_i['y_test'])
    a2 = flatten(results_i['y_test_pred'])
    axs[i].scatter(a1,a2 ,alpha=0.3,s=15, color = colors[i],marker='o', linewidth=1.5)
    axs[i].set_title(indeces[i], fontsize=fs+2, x=0.35, y=0.75)

    max_val_i = max(max(a1,a2))
    X_line = np.arange(0,max_val_i,max_val_i/200)
    axs[i].plot(X_line,X_line)

    if i == 3  or i == 0:
        axs[i].set_ylabel('Predicted values', fontsize=fs)
    axs[i].set_xlabel('Actual values', fontsize=12)

fig.suptitle('Scatter plot of predictions vs real values for different algorithms for the CPU Utilization (%)', fontsize=fs+2, x=0.5, y=0.93)
# plt.xticks( rotation=25 )

plt.savefig(os.path.join(sav_path,'scatter_plot.png'),bbox_inches='tight')

#%%
mean_test = np.argsort([np.mean(m_y) for m_y in results_i['y_test']])
var_test = np.argsort([np.var(m_y) for m_y in results_i['y_test']])
high_load_ind = mean_test[-1]
low_load_ind = mean_test[1]
high_var_ind = var_test[-1]
low_var_ind = var_test[0]
ind_plotMs = [high_load_ind,low_load_ind,high_var_ind,low_var_ind]
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
    a2 = flatten(results_i['y_test_pred'])
    train_time_all.append(results_i['train_time'])
    test_time_all.append(results_i['test_time'])
    
    RMSE_i = RMSE(a1,a2)
    MAE_i = MAE(a1,a2)
    MAPE_i = MAPE(a1,a2)
    data_res.append([RMSE_i,MAE_i,MAPE_i])


    plt.bar(counter , RMSE_i , color=colors[counter], width=barWidth, hatch="x", edgecolor='black', label = indeces[counter])

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontsize=fs)
plt.ylabel('RMSE', fontsize=fs)

plt.xticks([r for r in range(len(indeces_short))], indeces_short)
_,b = plt.ylim()
plt.ylim([0,b*1.1])
# yticks_no = np.arange(0,lim,steps)
# yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
# plt.yticks(yticks_no,yticks)
plt.title('Bar plot of the RMSE for different algorithms',fontsize=fs+2)
# Create legend & Show graphic
plt.legend(prop={'size': fs},loc=(0.05,0.5))
plt.gca().grid(True)
plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
plt.savefig(os.path.join(sav_path,'bar_plot.eps'),bbox_inches='tight', format='eps')

#%% train and test times
Metric = ['RMSE','MAE',"MAPE",'Train time (min)','Test time (s)']
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
TT = np.array([process_time(train_time_all),process_time(test_time_all)]).T
dat = np.concatenate((data_res, TT),axis=1)
df = pd.DataFrame(data=dat,columns=Metric,index = indeces)
print(df)
latex_txt = df.style.to_latex()


write_txt(latex_txt,os.path.join(sav_path,'results_table_latex.txt'))
#%% predictions vs real
# scale_mv = 460*52.5
fig, axs = plt.subplots(len(indeces),1, figsize=(20,11),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

# M_id = 880
#
append_test = []
for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    ind_machine_id = np.where(M_id_plot==np.array(results_i['Mids_test']))[0][0]
    len_i = len(np.squeeze(results_i['y_test'][ind_machine_id]))
    print(np.mean(results_i['y_test'][ind_machine_id]),np.shape(results_i['y_test'][ind_machine_id]))
    dummy = np.squeeze(results_i['y_test'][ind_machine_id])
    dummy = dummy[len(dummy)-122:]
    append_test.append(dummy)
    axs[i].plot(np.squeeze(results_i['y_test'][ind_machine_id]), 'o-', color = 'red', linewidth=1, alpha = 0.6)
    axs[i].plot(np.squeeze(results_i['y_test_pred'][ind_machine_id]), 'x-', color = 'blue', linewidth=0.8)
    axs[i].set_title(indeces[i], fontsize=fs-2, x=0.6, y=0.45)
    axs[i].set_xticks(np.arange(0,len_i,10) , 5*np.arange(0,len_i,10) )
    if i == 3:
        axs[i].set_ylabel('CPU utilization (%)', fontsize=fs+2)
    

axs[i].set_xlabel('Timestamp (min)' ,fontsize=fs)

#np.mean(np.array(append_test),axis=1)
# plt.legend(['Actual','Predicted'])
fig.legend(['Actual','Predicted'],prop={'size': fs},loc=(0.8,0.89))

M_id = np.array(results_i['Mids_test'])[ind_machine_id]
fig.suptitle('Prediction vs true values for machine number '+str(M_id), fontsize=fs, x=0.5, y=0.92)
# plt.xticks( np.arange(0,5,len_i*5) )

plt.savefig(os.path.join(sav_path,'PredictionsVsReal.eps'),bbox_inches='tight', format='eps')


#%%conv graph
# 

conv_data_path = os.path.join(working_path,'conv')
result_files = os.listdir(conv_data_path)
result_files = [file for file in result_files if file.endswith(".obj")]
algorithms_conv = ['CuckooSearch','Firefly']

indeces = [algorithms[c2] for c1,alg in enumerate(algorithms) for c2,file in enumerate(result_files) if alg in file]

# result_files = ['Best_paramMod_FireflyAlgorithm.obj','Best_paramFireflyAlgorithm.obj']

alg_rename_itr = {'Best_paramCuckooSearch':'Cuckoo Search',
              'Best_paramFireflyAlgorithm':'Fire Fly',
              'Best_paramMonkeyKingEvolutionV3':'MonkeyKingEvolutionV3'}

full_file_path2 = [os.path.join(conv_data_path,file) for file in result_files]

result_files = [file.split('.')[0] for file in result_files]

fig = plt.figure(figsize=(15,8),dpi=120)
# markers = ['o-','o-']
data_hp = []
for counter,res_file in enumerate(full_file_path2):
    results_i = loadDatasetObj(res_file)
    data_hp.append(list(results_i['best_para_save'].values()))
    # max_val = max(max_val,max(max(results_i['y_test']),max(results_i['y_test_pred'])))
    plt.plot(results_i['a_itr'],100* np.sqrt(results_i['b_itr']),'o-',label=algorithms_conv[counter], linewidth=3.0)

plt.xlabel('Iteration', fontsize=fs+2)
plt.ylabel('RMSE', fontsize=fs+2)
# plt.xlim(0)
plt.legend(prop={'size': fs+2})
# plt.ylim(0)
plt.show()
plt.title('Convergence Graph', fontsize=fs+2)
plt.gca().grid(True)
plt.savefig(os.path.join(sav_path,'Conv_eval_comparison.eps'),bbox_inches='tight', format='eps')




#%%
hp = list(results_i['best_para_save'].keys())
indeces_2 = [alg_rename_itr[f_i] for f_i in result_files]
df2 = pd.DataFrame(data=np.array(data_hp),columns=hp,index = indeces_2)
print(df2)


latex_txt_hp = df2.style.to_latex()

write_txt(latex_txt_hp,os.path.join(sav_path,'LSTM_HP_latex.txt'))
#%%