
# def get_ResNetPlus(trial):
#     ks_i = trial.suggest_categorical("ks", [3,5])
#     arch = dict(
#          nf = trial.suggest_categorical("nf", [32,64,128,256,512]),
#          ks= [ks_i,ks_i,ks_i],  # Example: Suggest different kernel size combinations
#          bn_1st= True,
#          sa = True,
#     )
#     return arch  

def get_FCNPlus(trial):
    kss_i = trial.suggest_categorical("kss", [3,5,7])
    layers_i = trial.suggest_categorical("layers", [32,64,128,256,512])
    arch_config = dict(
         layers=[layers_i, int(layers_i/2), int(layers_i/4)],
         kss= [kss_i,kss_i,kss_i],  # Example: Suggest different kernel size combinations
         use_bn= trial.suggest_categorical("use_bn", [True, False]),
         residual = trial.suggest_categorical("residual", [True, False]),
         fc_dropout = 0.5
    )
    return arch_config
    
    

def get_InceptionTimePlus(trial):

    arch_config = dict(
         nf = trial.suggest_categorical("nf", [32,64,128,256,512,1024]),
         ks= trial.suggest_categorical("ks", [3,5,7]),  # Example: Suggest different kernel size combinations
         bn_1st=trial.suggest_categorical("bn_1st", [True, False]),
         sa = True,
    )
    return arch_config
    

                 
def get_RNN_FCN(trial):
    kss_i = trial.suggest_categorical("kss", [3,5])
    conv_layer = trial.suggest_categorical("conv_layers", [32,64,128,256,512])
    arch_config = dict(
    hidden_size = trial.suggest_categorical("hidden_size", [32,64,128,256]),
    rnn_layers = trial.suggest_categorical("rnn_layers", [1,2,3]),
         conv_layers=[conv_layer, int(conv_layer/2), int(conv_layer/4)],
         kss= [kss_i,kss_i,kss_i],  # Example: Suggest different kernel size combinations
         fc_dropout = 0.25,
    )
    return arch_config
#%%
def get_best_FCNPlus(best_trial):
    kss_i = best_trial.params['kss']
    layers_i = best_trial.params["layers"]
    best_arch_config = dict(
         layers=[layers_i, layers_i*2, layers_i],
         kss= [kss_i,kss_i,kss_i],  # Example: Suggest different kernel size combinations
         use_bn= best_trial.params["use_bn"],
         residual = best_trial.params["residual"],
         fc_dropout = 0.1
    )
    return best_arch_config

def get_best_InceptionTimePlus(best_trial):

    best_arch_config = dict(
         nf = best_trial.params['nf'],
         ks= best_trial.params['ks'],  # Example: Suggest different kernel size combinations
         bn_1st= best_trial.params['bn_1st'],
         sa = True,
    )
    return best_arch_config

def get_best_RNN_FCN(trial):
    kss_i = best_trial.params['kss']
    conv_layer = best_trial.params['conv_layers']
    arch_config = dict(
    hidden_size = best_trial.params['hidden_size'],
    rnn_layers = best_trial.params['rnn_layers'],
         conv_layers=[conv_layer, int(conv_layer/2), int(conv_layer/4)],
         kss= [kss_i,kss_i,kss_i],  # Example: Suggest different kernel size combinations
         fc_dropout = 0.25,
    )
    return arch_config
