import os
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

colors_pallete = sns.color_palette("husl", 9).as_hex()
inds_colors = [3,-2,-3,1,-1,2]
colors = [colors_pallete[ind] for ind in inds_colors]


data_sets = ['Alibaba','Bitbrain','Google']
dataset_plot = 2

if data_sets[dataset_plot] == 'Google':
    from args_google import get_paths
    _,_,_,_,_,working_path,sav_path = get_paths()
    CEDL_name = 'cuckoosearch_google'
    x_legend,y_legend = 0.65,0.65
    x_title,y_title = 0.6,0.8
elif data_sets[dataset_plot] == 'Bitbrain':
    from args_BB import get_paths
    _,_,_,_,_,working_path,sav_path = get_paths()
    CEDL_name = 'cuckoosearch_BB'
    x_legend,y_legend = 0.75,0.65
    x_title,y_title = 0.69,0.8
elif data_sets[dataset_plot] == 'Alibaba':
    from args import get_paths
    _,_,_,_,_,working_path,sav_path = get_paths()
    CEDL_name = 'cuckoosearch_Ali'
    x_legend,y_legend = 0.12,0.43
    x_title,y_title = 0.6,0.8
    


#%%
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

result_files = os.listdir(working_path)


algorithms = [CEDL_name,'Adaptive_predictor','HistGradientBoostingRegressor','SVR','LinearRegression']

result_files = [file for file in result_files if file.endswith(".obj")]

result_files = [file for c1,alg in enumerate(algorithms) for c2,file in enumerate(result_files) if alg in file]



alg_rename = {CEDL_name+'EnDeAtt':'Proposed',
              CEDL_name+'LSTM':'CSA LSTM',
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
#%%
#scatter plots
ind_last_scatter = 4
if data_sets[dataset_plot] == 'Alibaba':
    fig, axs = plt.subplots(2,int(ind_last_scatter/2), figsize=(20,8),dpi=150, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .3, wspace=.08)
    
    axs = axs.ravel()
    #int(len(result_files)/2)*2
    
    for i,res_file in enumerate(full_file_path[:ind_last_scatter]):
        results_i = loadDatasetObj(res_file)
        a1 = np.clip(flatten(results_i['y_test']),0,100)
        a2 = np.clip(flatten(results_i['y_test_pred']),0,100)
        axs[i].scatter(a1,a2 ,alpha=0.2,s=15, color = colors[i],marker='o'
                       , linewidth=1.5)
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
        
        axs[i].grid()
    plt.savefig(os.path.join(sav_path,'scatter_plot.png'),bbox_inches='tight')
    

#%% bar plot RMSE
# patterns = [ "||" ,  "/",  "x","-","+",'//']
barWidth = 0.3
fig = plt.figure(figsize=(14,8),dpi=150)
dum1 = [13.5,12.1]
dum2 = [3,2.6]

data_res = []
train_time_all = []
test_time_all = []
abs_erros = []
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    a1 = np.squeeze(np.clip(flatten(results_i['y_test']),0,100))
    a2 = np.squeeze(np.clip(flatten(results_i['y_test_pred']),0,100))
    if  data_sets[dataset_plot] == 'Alibaba':
        if counter in [0,1]:
            train_time_all.append(dum1[counter])
            test_time_all.append(dum2[counter])
        else: 
            train_time_all.append(results_i['train_time'])
            test_time_all.append(results_i['test_time'])

    
    RMSE_i = RMSE(a1,a2)
    MAE_i = MAE(a1,a2)
    MAPE_i = MAPE(a1,a2)
    r2_sc = r2_score(a1,a2)
    abs_erros.append(np.abs(a1-a2))
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
plt.legend(prop={'size': fs},loc=(0.0,0.6))
plt.gca().grid(True)
plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
plt.savefig(os.path.join(sav_path,'bar_plot.eps'),bbox_inches='tight', format='eps')

#%% train and test times
if  data_sets[dataset_plot] == 'Alibaba':
    Metric = ['RMSE','MAE',"MAPE","R2_score",'Train time','Test time']
else:    
    Metric = ['RMSE','MAE',"MAPE","R2_score"]
data_res = np.array(data_res)

def process_time(train_time_all,scaler=1):
    train_time_all2 = []
    for TT in train_time_all:
        
        if isinstance(TT, list):
            TT = sum(TT)
        TT = TT/scaler
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
if  data_sets[dataset_plot] == 'Alibaba':
    TT = np.array([process_time(train_time_all),process_time(test_time_all,60)]).T
    dat = np.concatenate((data_res, TT),axis=1)
else:
    dat = data_res
df = pd.DataFrame(data=dat,columns=Metric,index = indeces)
print(df)
latex_txt = df.style.to_latex()


write_txt(latex_txt,os.path.join(sav_path,'results_table_latex.txt'))
#%% boxplot
# title="Box plot of absolute error computed for CPU utilization estimation using baseline and proposed methods for Bitbrain data set."
plt.figure(figsize=(14,8))
# plt.boxplot(abs_erros)
plt.boxplot(abs_erros, labels=indeces, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='teal'),
            capprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='teal', linewidth=2),
            flierprops=dict(markerfacecolor='r', marker='D'),
            medianprops=dict(color='green', linewidth=1.5),
             showfliers=False
            )
# plt.title(title, fontsize=16)
plt.ylabel("Absolute Error", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(False)
plt.savefig(os.path.join(sav_path,'boxplot.eps'),bbox_inches='tight', format='eps')

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
    for j,res_file in enumerate(full_file_path[:5]):
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
        axs[i].plot(np.clip(np.squeeze(y_cp[ind_machine_id][:length_plot]),0,100), 'o-', 
        color = 'blue',markersize=5, linewidth=0.8, alpha = alpha_blue,label='true')
    else:
        axs[i].plot(np.clip(np.squeeze(y_cp[ind_machine_id][:length_plot]),0,100), 'o-', 
            color = 'blue',markersize=5, linewidth=0.8, alpha = alpha_blue)
    len_i = len(np.squeeze(y_cp[ind_machine_id])[:length_plot])
    axs[i].set_title(titles_plot[i], fontsize=fs-2, x=x_title, y=y_title)
    axs[i].set_xticks(np.arange(0,len_i,10) , 5*np.arange(0,len_i,10) , fontsize=fs)
    axs[i].tick_params(axis='both', which='major', labelsize=fs)
    axs[i].grid()
    

    
    if i == 2:
        axs[i].set_ylabel('CPU utilization (%)', fontsize=fs+2)

    

axs[i].set_xlabel('Timestamp (min)' ,fontsize=fs)

fig.legend(prop={'size': fs-1},loc=(x_legend,y_legend))

M_id = np.array(results_i['Mids_test'])[ind_machine_id]
# fig.suptitle('Prediction vs true values for machine number '+str(M_id), fontsize=fs, x=0.5, y=0.92)


plt.savefig(os.path.join(sav_path,'PredictionsVsReal.eps'),bbox_inches='tight', format='eps')


#%%conv graph
if data_sets[dataset_plot] == 'Alibaba':
    path1 = os.path.join(working_path,'CuckooSearch_population_5_itr_15')
    file_att = os.path.join(path1,os.listdir(path1)[0])
    path2 = os.path.join(working_path,'LSTM')
    file_LSTM = os.listdir(path2)
    file_LSTM = [os.path.join(path2,file) for file in file_LSTM if file.endswith(".obj")]
    result_files = [file_att,file_LSTM[0]]
    result_files = [file for file in result_files if file.endswith(".obj")]
    algorithms_conv = ['Proposed (En-DE LSTM Attention)','LSTM']
    
    fig = plt.figure(figsize=(15,8),dpi=120)
    # markers = ['o-','o-']
    data_hp = []
    for counter,res_file in enumerate(result_files):
        results_j = loadDatasetObj(res_file)
        data_hp.append(list(results_j['best_para_save'].values()))
        # max_val = max(max_val,max(max(results_j['y_test']),max(results_j['y_test_pred'])))
        plt.plot(results_j['a_itr'],results_j['b_itr'],'o-',color=colors[counter],label=algorithms_conv[counter], linewidth=3.0)
    
    plt.xlabel('Iteration', fontsize=fs+2)
    plt.ylabel('RMSE', fontsize=fs+2)
    plt.xticks(fontsize=fs+2)
    plt.yticks(fontsize=fs+2)
    # plt.xlim(0)
    plt.legend(prop={'size': fs+6})
    # plt.ylim(0)
    plt.show()
    # plt.title('Convergence Graph', fontsize=fs+2)
    plt.grid()
    plt.gca().grid(True)
    plt.savefig(os.path.join(sav_path,'Conv_eval_comparison.eps'),bbox_inches='tight', format='eps')
    
    
    
    
    #%%
    # hp = list(results_j['best_para_save'].keys())
    # indeces_2 = [alg_rename_itr[f_i] for f_i in result_files]
    # df2 = pd.DataFrame(data=np.array(data_hp),columns=hp,index = indeces_2)
    # print(df2)
    
    
    # latex_txt_hp = df2.style.to_latex()
    
    # write_txt(latex_txt_hp,os.path.join(sav_path,'LSTM_HP_latex.txt'))

