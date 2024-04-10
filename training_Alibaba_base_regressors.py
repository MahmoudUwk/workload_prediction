
import numpy as np

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from Alibaba_helper_functions import RMSE,save_object,flatten,get_data_stat
import os
# from gpflow.mean_functions import Constant
# from gpflow.utilities import positive, print_summary

# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# import gpflow
import time
#%%
data_path = 'data/feature_statistical_proccessed/X_Y_alibaba_train_val_test_after_feature_removal.obj'
sav_path = 'data/models/regular_regressors'
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

RMSE_opt_all = []
test_set_len = []

X_train,y_train,X_test,y_test,scaler,df_test_xy = get_data_stat(data_path)
#%%


regs_all = [LinearRegression(), SVR(kernel= 'linear'),HistGradientBoostingRegressor()]

RMSE_local = []
rmse_list = []
Mids_test = []
y_test_pred_list = []
y_test_list = []
save_pred_save = 'data/pred_results_all'
for reg in regs_all:
    start_train = time.time()
    # if reg =='GPflow':
    #     k = gpflow.kernels.Matern52(input_dim=X_train.shape[1])
    #     m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)
    #     opt = gpflow.optimizers.Scipy()
    #     opt.minimize(m.training_loss, m.trainable_variables)
    #     print_summary(m)
    #     end_train = time.time()
    #     y_pred_opt, _ = m.predict_f(X_train)
    # else: 
    reg.fit(X_train, y_train)
    end_train = time.time()
    train_time = (end_train - start_train)/60
    
    
    y_pred_opt = reg.predict(X_test)
    
    
    rmse_i = RMSE(y_pred_opt,y_test)
    score_i = reg.score(X_test,y_test)
    x_test_all_list = []
    start_test = time.time()
    for m_id, group_val in  df_test_xy.groupby(["M_id"]):
        Mids_test.append(m_id)
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
    filename = os.path.join(save_pred_save,type(reg).__name__+'.obj')
    save_object(obj, filename)
    
    print("rmse_i:",rmse_i)
    print("np.mean(rmse_list):",RMSE(flatten(y_test_list),flatten(y_test_pred_list)))
    RMSE_local.append(rmse_i)
RMSE_opt_all.append(RMSE_local)
#%%
RMSE_opt_all = np.array(RMSE_opt_all)


print(RMSE_opt_all)











