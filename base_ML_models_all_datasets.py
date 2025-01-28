import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from utils_WLP import RMSE,save_object,flatten,get_adaptive_data
import os
from sklearn.svm import LinearSVR
# import gpflow
import time

data_set_flags = ['Alibaba','google','BB']
flag_datasets = [2]#[0,1,2]

for ind_DS in flag_datasets:
    flag_dataset = data_set_flags[ind_DS]
    if data_set_flags[ind_DS] == 'Alibaba':
        from args import get_paths
    elif data_set_flags[ind_DS]=='google':
        from args_google import get_paths
    elif data_set_flags[ind_DS]=='BB':
        from args_BB import get_paths
    base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()

    
    X_train,y_train,X_test,y_test,scaler,df_test_xy = get_adaptive_data(flag_dataset)
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
    #%%
    RMSE_opt_all = []
    test_set_len = []
    regs_all = [HistGradientBoostingRegressor()
                ,LinearRegression(),LinearSVR(max_iter=100000) ]
    
    RMSE_local = []
    for reg in regs_all:
        
        rmse_list = []
        Mids_test = []
        y_test_pred_list = []
        y_test_list = []
        start_train = time.time()
    
        reg.fit(X_train, y_train)
        end_train = time.time()
        train_time = (end_train - start_train)/60
        
        
        y_pred_opt = reg.predict(X_test)
        
        
        rmse_i = RMSE(y_pred_opt,y_test)
        # score_i = reg.score(X_test,y_test)
        x_test_all_list = []
        start_test = time.time()
        for m_id, group_val in  df_test_xy.groupby(["M_id"]):
            Mids_test.append(m_id[0])
            x_test_i = scaler.transform(np.array(group_val.drop(['y','M_id'],axis=1)))
            x_test_all_list.append(x_test_i)
            y_test_list.append(np.array(group_val['y']))
            pred_i = reg.predict(x_test_i)
            y_test_pred_list.append(pred_i)
            rmse_i_list = RMSE(np.array(group_val['y']),pred_i)
            rmse_list.append(rmse_i_list)
        end_test = time.time()
        test_time = end_test - start_test
        obj = {'test_time':test_time,'train_time':train_time,'y_test':y_test_list,'y_test_pred':y_test_pred_list,'rmse_list':np.array(rmse_list),'Mids_test':Mids_test}
        filename = os.path.join(sav_path,type(reg).__name__+'.obj')
        save_object(obj, filename)
        
        print("rmse_i:",rmse_i)
        print("np.mean(rmse_list):",RMSE(flatten(y_test_list),flatten(y_test_pred_list)))
        RMSE_local.append(rmse_i)
    RMSE_opt_all.append(RMSE_local)

    RMSE_opt_all = np.array(RMSE_opt_all)
    
    
    print(RMSE_opt_all)











