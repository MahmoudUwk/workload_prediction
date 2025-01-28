
def get_ResNetPlus(trial):
     ks_i = trial.suggest_categorical("ks", [3,5])
     arch = dict(
         nf = trial.suggest_categorical("nf", [64,128,256,512,1024]),
          ks= [ks_i,ks_i,ks_i],  # Example: Suggest different kernel size combinations
          bn_1st= trial.suggest_categorical("bn_1st", [True, False]),
          sa = trial.suggest_categorical("sa", [True, False]),
     )
     return arch  


def get_best_ResNetPlus(best_trial):
     ks_i = best_trial.params['ks']
     arch = dict(
         nf = best_trial.params['nf'],
          ks= [ks_i,ks_i,ks_i],  # Example: Suggest different kernel size combinations
          bn_1st= best_trial.params['bn_1st'],
          sa = best_trial.params['sa'],
     )
     return arch  





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
    
 

def get_XceptionModulePlus(trial):
    kss_i = trial.suggest_categorical("kss", [3,5])
    arch_config = dict(
         nf = 2**trial.suggest_int("nf", 8,9),
         kss= [kss_i,kss_i,kss_i],   # Example: Suggest different kernel size combinations
         bottleneck= True,#trial.suggest_categorical("bottleneck", [True, False]),
    )
    return arch_config
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_best_XceptionModulePlus(best_trial):
    kss_i = best_trial.params['kss']
    best_arch_config = dict(
         nf = 2**best_trial.params['nf'],
         kss=  [kss_i,kss_i,kss_i],  # Example: Suggest different kernel size combinations
         bottleneck= True,#best_trial.params['bottleneck'],
    )
    return best_arch_config



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
    


def get_best_RNN_FCN(best_trial):
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
    
def get_PatchTST(trial):
    n_heads = trial.suggest_int("n_heads", 4, 8)
    d_model_multiplier = 2**trial.suggest_int("d_model_multiplier", 6, 7)
    d_model = n_heads * d_model_multiplier 

    arch_config = dict(
        n_layers=trial.suggest_int("n_layers", 2, 4),
        n_heads=n_heads,
        d_model=d_model,
        d_ff= 2**trial.suggest_int("d_ff", 8, 10, log=True),
        attn_dropout=0,
        dropout=0.2,
        patch_len=trial.suggest_int("patch_len", 2, 26),
        stride=trial.suggest_int("stride", 1, 8),
        padding_patch=True,
    )
    return arch_config

def get_best_PatchTST(trial):
    # Reconstruct the architecture configuration from the best trial's parameters
    best_n_heads = trial.params["n_heads"] 
    best_d_model = best_n_heads *  2**trial.params["d_model_multiplier"]
    best_arch_config = dict(
        n_layers= trial.params["n_layers"], #4,#
        n_heads=best_n_heads,
        d_model=best_d_model,
        d_ff= trial.params["d_ff"], #512,#
        attn_dropout=0,
        dropout= 0.2,# You had a fixed dropout in the original code
        patch_len= trial.params["patch_len"], #12,#
        stride= trial.params["stride"],
        padding_patch=True,
    )
    return best_arch_config

def get_para_PatchTST():
    # Reconstruct the architecture configuration from the best trial's parameters
    best_n_heads = 4
    best_d_model = best_n_heads *  2**6
    best_arch_config = dict(
        n_layers= 3, #4,#
        n_heads=best_n_heads,
        d_model=best_d_model,
        d_ff= 256, #512,#
        attn_dropout=0,
        dropout= 0.2,# You had a fixed dropout in the original code
        patch_len= 17, #12,#
        stride= 3,
        padding_patch=True,
    )
    return best_arch_config
