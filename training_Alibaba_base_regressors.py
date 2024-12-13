
import numpy as np

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
# from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from Alibaba_helper_functions import RMSE,save_object,flatten,get_data_stat
import os
# from gpflow.mean_functions import Constant
# from gpflow.utilities import positive, print_summary

# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# import gpflow
import time
# %%
from args import get_paths
base_path,processed_path,_,_,feat_stats_step3,sav_path = get_paths()
data_path = feat_stats_step3
# sav_path = os.path.join(base_path, '/regular_regressors')
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

RMSE_opt_all = []
test_set_len = []
dat_path_obi = os.path.join(data_path,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
X_train,y_train,X_test,y_test,scaler,df_test_xy = get_data_stat(dat_path_obi)
#%%
# LinearSVR()
# SVR(kernel= 'linear', cache_size=4000, n_jobs=-1)
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
#%%
RMSE_opt_all = np.array(RMSE_opt_all)


print(RMSE_opt_all)











