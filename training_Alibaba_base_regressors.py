
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models_lib import reg_all,class_all
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from Alibaba_helper_functions import loadDatasetObj,save_object,diff,RMSE,expand_dims,expand_dims_st
import os
#%%
# base_path = "data/"
models = ["linear_reg","svr_reg","GBT_reg"]#,"GPR_reg"]
class_models_names = ["KNN","GNB","RDF","GBT","MLP"]
data_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/Proccessed_Alibaba'
sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/"
file_read  = os.listdir(data_path)
scaler_norm = 100
RMSE_opt_all = []
test_set_len = []
for M_id,file_M_id in enumerate(file_read):
    reg_trained_M_id = []
    # class_trained_M_id  = []
    print(M_id,file_M_id)
    filename = os.path.join(data_path,file_M_id)
    df = loadDatasetObj(filename)
    if 'M_id' in df['X_train'].columns:
        X_train = np.array(df['X_train'].drop(['M_id'],axis=1))
    else:
        X_train = np.array(df['X_train'])
    y_train = df['y_train']/scaler_norm
    
    if 'M_id' in df['X_train'].columns:
        X_test = np.array(df['X_test'].drop(['M_id'],axis=1))
    else:
        X_test = np.array(df['X_test'])
    
    y_test = df['y_test']/scaler_norm
    test_set_len.append(X_test.shape[0])
    print(X_train.shape,X_test.shape)
    #%%
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #%%
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    # regs_all = [RandomForestRegressor(max_depth=10),LinearRegression(), SVR(kernel= 'linear')]#, 
                #HistGradientBoostingRegressor()]
    regs_all = [RandomForestRegressor()]
    RMSE_local = []
    for reg in regs_all:
        reg.fit(X_train, y_train)
        y_pred_opt = reg.predict(X_test)
    
        
        RMSE_local.append(RMSE(y_pred_opt*scaler_norm,y_test*scaler_norm))
    RMSE_opt_all.append(RMSE_local)
#%%
RMSE_opt_all = np.array(RMSE_opt_all)
per = np.array(test_set_len)/sum(test_set_len)
per = np.expand_dims(per, axis=1)
# clusters= [459, 13, 205, 633]

RMSE_all =np.sum( np.array(RMSE_opt_all)*per,axis=0)

print(RMSE_all)











