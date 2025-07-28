import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils_WLP import write_txt,loadDatasetObj,RMSE,save_object,flatten,MAE,MAPE,get_alibaba_ids,shift_right
import pandas as pd
from sklearn.metrics import r2_score

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

#% bar plot RMSE
# patterns = [ "||" ,  "/",  "x","-","+",'//']
barWidth = 0.3
fig = plt.figure(figsize=(14,8),dpi=150)


data_res = []
train_time_all = []
test_time_all = []
abs_erros = []
M_ids_all = []
for counter,res_file in enumerate(full_file_path):
    
    results_i = loadDatasetObj(res_file)
    # M_ids_all.append(results_i['Mids_test'])
    scal1 = 1
    if np.max(results_i['y_test'][0])<2:
        scal1 = 100
    # print(scal1)

    a1 = np.squeeze(scal1*flatten(results_i['y_test']))
    a2 = np.squeeze(flatten(results_i['y_test_pred']))
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


plt.ylabel('RMSE', fontsize=fs)


plt.xticks([r for r in range(len(indeces_short))], indeces_short, fontsize=fs-3)
_,b = plt.ylim()
# plt.ylim([0,b*1.1])

plt.legend(prop={'size': fs-1},loc=(0.8,0.2))
plt.gca().grid(True)
plt.show()

plt.savefig(os.path.join(sav_path,'bar_plot_scheduling.eps'),bbox_inches='tight', format='eps')
plt.savefig(os.path.join(sav_path,'bar_plot_scheduling.png'),bbox_inches='tight', format='png')

# train and test times
# if  data_sets[dataset_plot] == 'Alibaba':
#     Metric = ['RMSE','MAE',"MAPE","$R^2$",'Train time','Test time']
# else:    
#     Metric = ['RMSE','MAE',"MAPE","$R^2$"]
Metric = ['RMSE','MAE',"MAPE(\%)","$R^2$",'Train time','Test time']
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

TT = np.array([process_time(train_time_all),process_time(test_time_all,60)]).T

df_tt = pd.DataFrame(data=TT, columns = Metric[-2:], index = indeces) # Using the first two metrics
df_res = pd.DataFrame(data=data_res, columns = Metric[:-2], index = indeces).round(2) # Using the last four metrics

# 2. Concatenate the DataFrames
df = pd.concat([df_res, df_tt], axis=1).round(2)

#%%
min_cols = Metric[:3]
s = df.style.highlight_min(subset=min_cols,axis=0, 
    props='cellcolor:[HTML]{E0E0E0}; color:{blue}; itshape:; bfseries:;'
)
max_cols = Metric[3]
s = s.highlight_max(subset=max_cols,axis=0, 
    props='cellcolor:[HTML]{E0E0E0}; color:{blue}; itshape:; bfseries:;'
)
s=s.format( precision=2)
latex_txt = s.to_latex(
    # float_format="%.2f" ,
    column_format="|m{1.3cm}|m{0.7cm}|m{0.5cm}|m{1.1cm}|m{0.4cm}|m{1cm}|m{1cm}|", 
    position="h", position_float="centering",

    hrules=True, label=table_label, caption="Results for the "+dataset_names_label[dataset_plot] +" testset.",

    multirow_align="t", multicol_align="r"

)  
print(data_res)
write_txt(latex_txt,os.path.join(sav_path,dataset_names_label[dataset_plot]+'results_table_latex.tex'))
#%% boxplot
# title="Box plot of absolute error computed for CPU utilization estimation using baseline and proposed methods for Bitbrain data set."
plt.figure(figsize=(14,8))
# plt.boxplot(abs_erros)
boxplot_data  = plt.boxplot(abs_erros, labels=indeces, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='teal'),
            capprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='teal', linewidth=2),
            flierprops=dict(markerfacecolor='r', marker='D'),
            medianprops=dict(color='green', linewidth=1.5),
             showfliers=False
            )
# plt.title(title, fontsize=16)
plt.ylabel("Absolute Error", fontsize=fs)
plt.grid()
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(False)
plt.savefig(os.path.join(sav_path,'boxplot.eps'),bbox_inches='tight', format='eps')

relevant_data = []
for i in range(len(abs_erros)):
    # Extract Q1 and Q3 from the box vertices
    box_path = boxplot_data['boxes'][i].get_path()
    vertices = box_path.vertices
    q1 = vertices[0, 1]  # Y-coordinate of the first corner (Q1)
    q3 = vertices[2, 1]  # Y-coordinate of the third corner (Q3)

    # Extract median, whiskers
    median = boxplot_data['medians'][i].get_ydata()[0]
    whisker_low = boxplot_data['whiskers'][2 * i].get_ydata()[1]
    whisker_high = boxplot_data['whiskers'][2 * i + 1].get_ydata()[1]

    # Collect data
    data = {
        "label": indeces[i],
        "median": median,
        "q1": q1,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }
    relevant_data.append(data)
#%% times series plot
fs = 15
def get_ind_plot(y,i):
    y= results_i['y_test']
    m_ids_plot = results_i['Mids_test']
    if isinstance(m_ids_plot[0],tuple):
        m_ids_plot = [id_i[0]  for id_i in m_ids_plot]
    mean_test = np.argsort([np.mean(100*m_y) for m_y in y])
    var_test = np.argsort([np.var(100*m_y) for m_y in y])
    high_load_ind = mean_test[ipTS[0]]
    low_load_ind = mean_test[ipTS[1]]  #alibaba inde -1 1 -1 2
    high_var_ind = var_test[ipTS[2]]
    low_var_ind = var_test[ipTS[3]]
    ind_plotMs = [high_load_ind,low_load_ind,high_var_ind,low_var_ind]
    return ind_plotMs[i], m_ids_plot[ind_plotMs[i]]

def append_arr(y_true,y_pred):
    return np.concatenate([y_true[:(len(y_true)-len(y_pred))],y_pred],axis=0)

length_plot = 100

titles_plot = ['High Load','Low Load', 'High Variation', 'Low Variation']
fig, axs = plt.subplots(len(titles_plot),1, figsize=(18,9),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .3, wspace=.001)
axs = axs.ravel()
alpha_blue = 0.35
markers = ['x','','','','']
results_i = loadDatasetObj(full_file_path[2])
y_cp = results_i['y_test'].copy()
for i,title_i in enumerate(titles_plot):
    ind_plain , ind_machine_id = get_ind_plot(results_i,i)
    y_cp_i = y_cp[ind_plain]
    scal1 = 1
    if np.max(y_cp_i)<2:
        y_cp_i = 100*y_cp_i
    for j,res_file in enumerate(full_file_path[:4]):

        results_i = loadDatasetObj(res_file)
        M_id_test_i = results_i['Mids_test']

        if isinstance(M_id_test_i[0],tuple):
            M_id_test_i = [id_i[0]  for id_i in M_id_test_i]

        print('counter',j)

        
        id_pred = [c for c,id_i in enumerate(M_id_test_i) if id_i==ind_machine_id][0]
        
        print(indeces[j],ind_machine_id)
        y_pred_plot = append_arr(y_cp_i,
                         np.squeeze(results_i['y_test_pred'][id_pred]))[spInd:spInd+length_plot]
        y_pred_plot = np.clip(y_pred_plot,0,100)
        if i == 3:
            axs[i].plot(y_pred_plot, markers[j]+'-', 
                        color = colors[j],alpha=0.6, linewidth=0.8,label=indeces[j])
        else:
            axs[i].plot(y_pred_plot, markers[j]+'-', 
                        color = colors[j],alpha=0.6, linewidth=0.8)
    

    if i == 3:
        axs[i].plot(np.clip(np.squeeze(y_cp_i[spInd:spInd+length_plot]),0,100), 'o--', 
        color = 'm',markersize=3, linewidth=0.5, alpha = alpha_blue,label='true values')
    else:
        axs[i].plot(np.clip(np.squeeze(y_cp_i[spInd:spInd+length_plot]),0,100), 'o--', 
            color = 'm',markersize=3, linewidth=0.5, alpha = alpha_blue)
    len_i = len(np.squeeze(y_cp_i)[spInd:spInd+length_plot])
    axs[i].set_title(titles_plot[i], fontsize=fs-2, x=x_title, y=y_title)
    axs[i].set_xticks(np.arange(0,len_i,10) , 5*np.arange(0,len_i,10) , fontsize=fs-2)
    axs[i].tick_params(axis='both', which='major', labelsize=fs-2)
    axs[i].grid()
    

    
    if i == 1:
        axs[i].set_ylabel('CPU utilization (%)', fontsize=fs-2)

    

axs[i].set_xlabel('Timestamp (min)' ,fontsize=fs-2)

fig.legend(prop={'size': fs-2},loc=(x_legend,y_legend))

# M_id = np.array(results_i['Mids_test'])[ind_machine_id]
# fig.suptitle('Prediction vs true values for machine number '+str(M_id), fontsize=fs, x=0.5, y=0.92)


plt.savefig(os.path.join(sav_path,'PredictionsVsReal.eps'),bbox_inches='tight', format='eps')

#%%
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
    

#%%conv graph
fs = 18
if data_sets[dataset_plot] == 'Alibaba':
    path1 = os.path.join(working_path,'CuckooSearch_population_5_itr_15')
    file_att = os.path.join(path1,os.listdir(path1)[0])
    path2 = os.path.join(working_path,'LSTM')
    file_LSTM = os.listdir(path2)
    file_LSTM = [os.path.join(path2,file) for file in file_LSTM if file.endswith(".obj")]
    path0 = os.path.join(working_path,'tempoSight')
    file_tempoSight = os.path.join(path0,os.listdir(path1)[0])
    result_files = [file_tempoSight,file_att,file_LSTM[0]]
    result_files = [file for file in result_files if file.endswith(".obj")]
    algorithms_conv = ['TempoSight','En-De LSTM attention','LSTM']
    
    fig = plt.figure(figsize=(15,8),dpi=120)
    # markers = ['o-','o-']
    data_hp = []
    for counter,res_file in enumerate(result_files):
        results_j = loadDatasetObj(res_file)
        data_hp.append(list(results_j['best_para_save'].values()))
        # max_val = max(max_val,max(max(results_j['y_test']),max(results_j['y_test_pred'])))
        plt.plot(results_j['a_itr'],results_j['b_itr'],'o-',color=colors[counter],label=algorithms_conv[counter], linewidth=3.0)
    
    plt.xlabel('Iteration', fontsize=fs+4)
    plt.ylabel('RMSE', fontsize=fs+4)
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

