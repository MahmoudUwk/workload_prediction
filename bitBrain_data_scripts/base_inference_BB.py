#BB data inference
import numpy as np
import os
import sys
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from Alibaba_helper_functions import RMSE,save_object,flatten,get_data_stat,loadDatasetObj
from sklearn.svm import SVR,LinearSVR
import time
from args import get_paths
_,_,_,_,feat_stats_step3,_,_ = get_paths()
dat_path_obi = os.path.join(feat_stats_step3,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
X_train,y_train,_,_,scaler,_ = get_data_stat(dat_path_obi)



from args_BB import get_paths
base_path,processed_path,feat_BB_step1,feat_BB_step2,feat_BB_step3,sav_path,sav_path_plots = get_paths()

if not os.path.exists(sav_path):
    os.makedirs(sav_path)
    
dat_BB_obi = os.path.join(feat_BB_step3,'XY_test_ready.obj')

df_test_xy = loadDatasetObj(dat_BB_obi)['XY_test_ready']
id_m = 'M_id'
X_test = scaler.transform(np.array(df_test_xy.drop(['M_id','y'],axis=1)))
y_test  = np.array(df_test_xy['y'])


RMSE_opt_all = []
test_set_len = []
#%%
regs_all = [LinearRegression() , 
            HistGradientBoostingRegressor(),LinearSVR(max_iter=100000) ]

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
    
    
    y_pred_opt = np.clip(reg.predict(X_test),0,100)
    
    
    rmse_i = RMSE(y_pred_opt,y_test)
    # score_i = reg.score(X_test,y_test)
    x_test_all_list = []
    start_test = time.time()
    for m_id, group_val in  df_test_xy.groupby([id_m]):
        Mids_test.append(m_id[0])
        x_test_i = scaler.transform(np.array(group_val.drop(['y','M_id'],axis=1)))
        x_test_all_list.append(x_test_i)
        y_test_list.append(np.array(group_val['y']))
        pred_i = np.clip(reg.predict(x_test_i),0,100)
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
#%%
RMSE_opt_all = np.array(RMSE_opt_all)


print(RMSE_opt_all)











