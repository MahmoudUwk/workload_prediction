# from models_lib import reg_all
import time
import os
import numpy as np
from utils_WLP import reg_all,RMSE,save_object,flatten,get_adaptive_data,class_all,expand_dims_st

class_models_names = ["KNN","GNB","RDF","GBT","MLP"]
models = ["linear_reg","svr_reg","GBT_reg"]
data_set_flags = ['Alibaba','google','BB']
flag_datasets = [2]



for ind_DS in flag_datasets:
    flag_dataset = data_set_flags[ind_DS]
    if data_set_flags[ind_DS] == 'Alibaba':
        from args import get_paths
    elif data_set_flags[ind_DS]=='google':
        from args_google import get_paths
    elif data_set_flags[ind_DS]=='BB':
        from args_BB import get_paths
    base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
    
    X_train,y_train,X_test,y_test,scaler,df_test_xy = get_adaptive_data(flag_dataset)
    pred_rr_train = np.zeros((len(X_train),len(models)))
    pred_rr_test = np.zeros((len(X_test),len(models)))

    reg_trained_all = []
    train_time_all = []
    for model_counter , model in enumerate(models):
        start_train = time.time()
        reg_trained = reg_all(X_train,y_train,X_test,model)
        end_train = time.time()
        train_time = (end_train - start_train)/60
        train_time_all.append(train_time)
        
        y_pred_train = reg_trained.predict(X_train)
        y_pred_test = reg_trained.predict(X_test)
        pred_rr_train[:,model_counter] = np.abs(y_pred_train-y_train)
        pred_rr_test[:,model_counter] = np.abs(y_pred_test-y_test)
        reg_trained_all.append(reg_trained)
    #%%
    y_train_c = np.argmin(pred_rr_train,axis=1)
    y_test_c = np.argmin(pred_rr_test,axis=1)
    
    acc_c = np.zeros((len(class_models_names)))
    y_pred_all = []
    class_trained_all = []
    for model_counter , model in enumerate(class_models_names):
        print(model)
        start_train = time.time()
        classifier_trained,y_pred = class_all(X_train,y_train_c,X_test,model)
        end_train = time.time()
        train_time_all.append((end_train - start_train)/60)
        y_pred_all.append(y_pred)
        # print(y_pred.round(2))
        acc_c[model_counter]= np.mean(y_pred==y_test_c)
        class_trained_all.append(classifier_trained)

    ind_best_classifier = np.argmax(acc_c)
    print(acc_c)
    print('Best classifier:',class_models_names[ind_best_classifier])
    #%%
    class_trained_best = class_trained_all[ind_best_classifier]
    y_pred_regressor = y_pred_all[ind_best_classifier]
    y_reg_adaptive = np.zeros((len(X_test)))
    RMSE_opt_all = []
    for c_i , test_instance in enumerate(X_test):
        y_reg_adaptive[c_i] = reg_trained_all[y_pred_regressor[c_i]].predict(expand_dims_st(test_instance))[0]

    RMSE_opt_all = RMSE(y_reg_adaptive,y_test)
    #%%
    Mids_test = []
    y_test_list = []
    y_test_pred_list = []
    rmse_list = []
    start_test = time.time()
    for m_id, group_val in  df_test_xy.groupby(["M_id"]):
        Mids_test.append(m_id[0])
        y_test_list.append(np.array(group_val['y']))
        X_test_Mid = scaler.transform(np.array(group_val.drop(['y','M_id'],axis=1)))
        y_pred_reg_best = []
        ind_regs = class_trained_best.predict(X_test_Mid)
        for counter,test_instance in enumerate(X_test_Mid):
            ind_reg = ind_regs[counter]
            y_i = reg_trained_all[ind_reg].predict(expand_dims_st(test_instance))[0]
            y_pred_reg_best.append(y_i)
        y_test_pred_list.append(y_pred_reg_best)
        rmse_i_list = RMSE(np.array(group_val['y']),y_pred_reg_best)
        rmse_list.append(rmse_i_list)
    end_test = time.time()
    test_time = end_test - start_test
    obj = {'class_models_names':class_models_names,'acc_c':acc_c,'test_time':test_time,'train_time':train_time_all,'y_test':y_test_list,'y_test_pred':y_test_pred_list,'rmse_list':np.array(rmse_list),'Mids_test':Mids_test,'Best classifier':class_models_names[ind_best_classifier]}
    filename = os.path.join(sav_path,'Adaptive_predictor.obj')
    save_object(obj, filename)
    print(RMSE_opt_all)
    print('RMSE:',np.mean(rmse_list))
    print("RMSE:",RMSE(flatten(y_test_list),flatten(y_test_pred_list)))
    

    
    












