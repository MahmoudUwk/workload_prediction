import pandas as pd
import numpy as np
import pickle
import os
from keras.optimizers import Adam
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.losses import Huber
#%%

def shift_right(lst, n):
    n = n % len(lst)
    return lst[-n:] + lst[:-n]

def write_txt(txt,fname):
    f = open(fname, "w")
    f.write(txt)
    f.close()
    
    
def flatten(xss):
    return [x for xs in xss for x in xs]

def drop_col_nan_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(axis=1)


def list_to_array(lst):
    if lst[0].ndim==3:
        _,seq_length,n_feat = lst[0].shape
        flag = 0
    else:
        flag = 1
    shapes = 0
    for sub_list in lst:
        shapes += len(sub_list)
    ind = 0
    if flag == 0:
        X = np.zeros((shapes,seq_length,n_feat))
        for sub_list in lst:
            X[ind:ind+len(sub_list),:,:] = sub_list
            ind = ind + len(sub_list)
    else:
        X = np.zeros((shapes,))
        for sub_list in lst:
            X[ind:ind+len(sub_list)] = sub_list
            ind = ind + len(sub_list)
  
    return X

def split_3d_array(array_3d, batch_size):
    num_samples = array_3d.shape[0]
    sub_arrays = []
    for i in range(0, num_samples, batch_size):
        sub_array = array_3d[i:i + batch_size]
        sub_arrays.append(sub_array)
    return sub_arrays

def diff(test,pred):
    return np.abs(test-pred)


def expand_dims_st(X):
    return np.expand_dims(X, axis = 0)


def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))


def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(np.array(test)) - np.squeeze(np.array(pred)))**2))

def MAE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))

def MAPE(test,pred):
    test = np.array(test)
    pred = np.array(pred)
    ind = np.where(test!=0)[0].flatten()
    return 100*np.mean(np.abs(np.squeeze(pred[ind]) - np.squeeze(test[ind]))/np.abs(np.squeeze(test[ind])))


def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def df_from_M_id(df,M):
    return df.loc[df["id"].isin(M)]

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
        
def sliding_window_i(data, seq_length,target_col_num):
    
    x = np.zeros((len(data)-seq_length,seq_length,data.shape[1]))
    y = np.zeros((len(data)-seq_length))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        x[ind,:,:] = data[ind:ind+seq_length,:]
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind] = data[ind+seq_length:ind+seq_length+1,target_col_num][0]
    return x,y
    
def window_rolling(df,seq_length):
    cols = ['y','x1']
    target = 'y'
    X = []
    Y = []
    M_ids = []
    for M_id, M_id_values in df.groupby(["id"]):
        M_id_values = M_id_values.sort_values(by=['date']
                  ).reset_index(drop=True).drop(["id","date"],axis=1)[cols]
        
        target_col_num = [ind for ind,col in enumerate(list(M_id_values.columns)) if col==target][0]
    
        X_train,y_train = sliding_window_i(np.array(M_id_values), seq_length,target_col_num)
        X.append(X_train)
        Y.append(y_train)
        M_ids.append(M_id)
  
    return X,Y,M_ids

#%%
def get_lstm_model(input_shape, output_dim, units, num_layers, dense_units, lr=0.001, seq=12, name='LSTM_Model'):
    from keras.layers import Dense, LSTM, Input
    from keras.models import Model
    


    # Input Layer
    inputs = Input(shape=input_shape, name='input')

    # LSTM Layers
    lstm_outputs = inputs  # Initialize with the input
    for i in range(num_layers):
        lstm_outputs = LSTM(units, return_sequences=(i < num_layers - 1),
                           name=f"lstm_{i}")(lstm_outputs)

    # Dense Layers
    outputs = Dense(dense_units, activation='relu', name="dense_1")(lstm_outputs)
    outputs = Dense(output_dim, activation=None, name="output_layer")(outputs)

    # Model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1))
    return model
#%% LSTM models

#swish relu gelu

def get_en_de_lstm_model_attention(input_shape,output_dim, units, num_layers,dense_units, lr=0.005, seq=12,
       name='EncoderDecoderLSTM_MTO_Attention'):
    from keras.layers import Dense,Flatten, LSTM, Input,Dropout, AdditiveAttention, concatenate, RepeatVector
    from keras.models import Model
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from keras.optimizers import Adam
    from tensorflow.keras.losses import Huber
    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    encoder_outputs = LSTM(units, return_sequences=True)(encoder_inputs)
    for i in range(num_layers - 2):
        encoder_outputs = LSTM(units, return_sequences=True, name=f"encoder_lstm_{i+1}")(encoder_outputs)
    
    # Final encoder layer returns both sequences and states
    encoder_outputs, state_h, state_c = LSTM(units, return_sequences=True, return_state=True, name=f"encoder_lstm_{num_layers-1}")(encoder_outputs)
    encoder_states = [state_h, state_c]

    # Decoder
    # RepeatVector to expand the final encoder state to match the expected decoder input
    context = RepeatVector(1, name="context_vector")(encoder_outputs[:, -1, :])

    # Decoder layers
    decoder = LSTM(units, return_sequences=True, name="decoder_lstm_0")(context, initial_state=encoder_states)
    for i in range(num_layers - 1):
        decoder = LSTM(units, return_sequences=True, name=f"decoder_lstm_{i+1}")(decoder)

    # Attention Mechanism
    attention_layer = AdditiveAttention(name="attention_layer")
    attention_output = attention_layer([decoder, encoder_outputs])

    # Concatenate attention output and decoder output
    decoder_combined_context = concatenate([decoder, attention_output], axis=-1, name="concatenate_output_attention")

    # Dense layers for final prediction
    outputs = Dense(dense_units, activation='swish', name="dense_1")(decoder_combined_context)
    # outputs = Dropout(0.2)(outputs)
    outputs = Dense(output_dim, activation=None, name="output_layer")(outputs)

    # Flatten the output
    outputs = Flatten(name="flatten_output")(outputs)

    # Create and compile model
    model = Model(inputs=encoder_inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1.35))
    

    return model



#%%import numpy as np
#version V1 above is the one used
def get_en_de_lstm_model_attentionV2(input_shape, output_dim, units, num_layers, dense_units
         , lr=0.002, seq=12, num_heads=8, name='EncoderDecoderLSTM_MTO_AttentionV2'):
    from keras.layers import Dense, Flatten, LSTM, Input, concatenate, RepeatVector, LayerNormalization
    from keras.models import Model
    from keras.optimizers import Adam
    
    from tensorflow.keras.layers import MultiHeadAttention
    

    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    encoder_outputs = LSTM(units, return_sequences=True)(encoder_inputs)
    for i in range(num_layers - 2):
        encoder_outputs = LSTM(units, return_sequences=True, name=f"encoder_lstm_{i+1}")(encoder_outputs)
    
    # Final encoder layer returns both sequences and states
    encoder_outputs, state_h, state_c = LSTM(units, return_sequences=True, return_state=True, name=f"encoder_lstm_{num_layers-1}")(encoder_outputs)
    encoder_states = [state_h, state_c]

    # Decoder
    # RepeatVector to expand the final encoder state to match the expected decoder input
    context = RepeatVector(1, name="context_vector")(encoder_outputs[:, -1, :])

    # Decoder layers
    decoder = LSTM(units, return_sequences=True, name="decoder_lstm_0")(context, initial_state=encoder_states)
    for i in range(num_layers - 1):
        decoder = LSTM(units, return_sequences=True, name=f"decoder_lstm_{i+1}")(decoder)

    # Multi-Head Attention Mechanism
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=units, name="multi_head_attention")(query=decoder, value=encoder_outputs, key=encoder_outputs)

    # Concatenate attention output and decoder output
    decoder_combined_context = concatenate([decoder, attention_output], axis=-1, name="concatenate_output_attention")

    # Dense layer to match the dimensions
    # decoder_combined_context = Dense(units, activation='swish', name="dense_match_units")(decoder_combined_context)

    # Add Layer Normalization
    # decoder_combined_context = LayerNormalization(epsilon=1e-6, name="layer_norm_1")(decoder_combined_context + decoder)

    # Dense layers for final prediction
    outputs = Dense(dense_units, activation='swish', name="dense_1")(decoder_combined_context)
    outputs = Dense(output_dim, activation=None, name="output_layer")(outputs)

    # Flatten the output
    outputs = Flatten(name="flatten_output")(outputs)

    # Create and compile model
    model = Model(inputs=encoder_inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1))

    # Callbacks
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-4)

    return model, reduce_lr

#%% Alibaba dataset adaptive preprocessing

def get_data_stat(data_path):
    def from_M_id(df,M):
        return df.loc[df["M_id"].isin(M)]

    M_ids_train, M_ids_val, M_ids_test = get_alibaba_ids()
    train_ids = M_ids_train+M_ids_val
    test_ids = M_ids_test
    df = loadDatasetObj(data_path)

    df_appended = pd.concat([df['XY_train'], df['XY_test']], ignore_index=True)
    del df
    df_train = from_M_id(df_appended, train_ids)
    
    df_test = from_M_id(df_appended, test_ids)

    X_train = np.array(df_train.drop(['M_id','y'],axis=1))

    y_train = np.array(df_train['y'])


    X_test = np.array(df_test.drop(['M_id','y'],axis=1))


    y_test = np.array(df_test['y'])
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train,y_train,X_test,y_test,scaler,df_test


#%% Alibaba dataset split ids train val test

def shuffle_with_seed(data):
    seed = 7
    import random
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    return shuffled

def get_train_test_Mids(df,train_val_per,val_per):
    M_ids_unshuffled = list(df["id"].unique())
    M_ids = shuffle_with_seed(M_ids_unshuffled)
    train_per = train_val_per - val_per
    
    train_len  = int(train_per * len(M_ids))
    val_len = int(val_per * len(M_ids))
    return M_ids[:train_len],M_ids[train_len:train_len+val_len],M_ids[train_len+val_len:]



# 1. Data Loading and Preprocessing Using get_df_tft
def prepare_data_for_tft(df,rename_map):
    

    reference_time = datetime(2017, 1, 1)
    df[" timestamp"] = df[" timestamp"].apply(lambda x: reference_time + timedelta(seconds=x))
    
    df = df.sort_values([' machine id', ' timestamp'])
    
    df.dropna(inplace=True)
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df
def get_df_alibaba():
    from args import get_paths
    base_path, _, _, _, _, _, _ = get_paths()
    script = "server_usage.csv"
    info_path = base_path + "schema.csv"
    df_info = pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']
    full_path = base_path + script
    nrows = None
    df = pd.read_csv(
        full_path, nrows=nrows, header=None, names=list(df_info)
    )
    
    df.dropna(inplace=True)
    normalize_cols = ['y', 'x1']
    rename_map = {
        ' used percent of cpus(%)': "y",
        ' used percent of memory(%)': "x1",
        ' machine id': "id",
        ' timestamp':'date',
    }
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df = df.loc[:,list(rename_map.values())]
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler
    
    reference_time = datetime(2017, 1, 1)
    df["date"] = df["date"].apply(lambda x: reference_time + timedelta(seconds=x))
    
    df = df.sort_values(['id', 'date'])
    

    df.reset_index(drop=True, inplace=True)
    
    # train_val_per = 0.8
    # val_per = 0.16

    # M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
    #     df,train_val_per ,val_per)
    M_ids_train, M_ids_val, M_ids_test = get_alibaba_ids()

    df_train = df_from_M_id(df, M_ids_train)
    df_val = df_from_M_id(df, M_ids_val)
    df_test = df_from_M_id(df, M_ids_test)

    return df_train, df_val, df_test

def get_alibaba_ids():
    M_ids_train = [539, 457, 1209, 112, 839, 1210, 1202, 389, 1012, 265, 877, 893, 326, 876, 651, 373, 275, 369, 173, 614, 347, 870, 963, 55, 844, 46, 864, 937, 323, 1141, 513, 824, 485, 1182, 1306, 733, 130, 481, 1175, 664, 576, 234, 1018, 901, 451, 833, 653, 264, 330, 340, 621, 631, 74, 1228, 358, 180, 502, 934, 659, 1113, 899, 897, 908, 329, 635, 1131, 942, 698, 316, 1016, 693, 1046, 1214, 490, 775, 506, 89, 1049, 33, 176, 771, 802, 1252, 523, 239, 673, 158, 399, 717, 1308, 424, 557, 1142, 199, 768, 1263, 23, 356, 1106, 10, 772, 303, 218, 758, 670, 391, 628, 408, 413, 618, 1279, 924, 847, 221, 615, 910, 397, 14, 988, 305, 174, 604, 563, 183, 936, 1072, 377, 1053, 407, 925, 1000, 851, 49, 794, 40, 560, 610, 558, 980, 1267, 182, 542, 235, 692, 1154, 1167, 642, 993, 76, 1006, 1226, 1183, 978, 929, 594, 804, 735, 307, 378, 445, 1171, 54, 22, 281, 606, 960, 1050, 715, 396, 1127, 140, 789, 1215, 795, 892, 1070, 533, 1260, 632, 882, 394, 497, 388, 1116, 819, 118, 470, 84, 61, 1057, 483, 1235, 1313, 1233, 3, 461, 920, 34, 809, 1311, 1265, 366, 414, 1103, 1084, 251, 1026, 1082, 7, 1256, 565, 967, 1246, 890, 436, 361, 581, 71, 82, 1112, 185, 475, 770, 1190, 489, 592, 70, 580, 860, 498, 972, 786, 478, 379, 954, 705, 1051, 315, 1166, 895, 871, 919, 425, 291, 611, 822, 320, 355, 8, 597, 970, 756, 1040, 484, 535, 761, 192, 959, 412, 873, 1240, 421, 685, 787, 995, 482, 977, 879, 181, 1296, 779, 534, 552, 1312, 911, 228, 1176, 1216, 195, 762, 456, 567, 381, 1229, 98, 170, 85, 854, 1269, 162, 589, 587, 638, 31, 27, 504, 1134, 916, 110, 28, 709, 828, 343, 215, 512, 363, 1280, 319, 868, 153, 1068, 114, 1244, 1304, 553, 1099, 103, 1014, 106, 35, 1271, 1237, 596, 607, 204, 799, 722, 442, 1238, 254, 1009, 909, 1184, 117, 132, 42, 582, 711, 439, 100, 419, 955, 159, 845, 948, 1126, 1286, 1248, 1199, 724, 245, 1178, 382, 728, 438, 662, 624, 585, 702, 671, 309, 682, 1242, 200, 467, 1165, 446, 903, 37, 640, 1029, 359, 716, 1047, 430, 1261, 322, 598, 1170, 368, 1258, 649, 950, 120, 435, 549, 21, 284, 961, 1234, 1007, 360, 578, 1211, 351, 269, 78, 620, 115, 750, 402, 426, 889, 1201, 1004, 547, 1179, 1138, 796, 732, 244, 989, 365, 912, 665, 627, 814, 418, 161, 223, 230, 1274, 1277, 449, 1148, 744, 622, 1158, 406, 324, 648, 1160, 353, 951, 186, 946, 669, 840, 660, 556, 1048, 139, 1307, 657, 956, 1032, 450, 198, 1268, 447, 385, 11, 104, 293, 752, 755, 190, 493, 165, 898, 286, 605, 90, 1083, 816, 1310, 1285, 865, 220, 232, 390, 172, 1283, 233, 1193, 680, 1114, 1264, 941, 783, 107, 39, 667, 331, 710, 62, 203, 1097, 1076, 249, 156, 1157, 337, 926, 999, 836, 812, 380, 131, 696, 1019, 1117, 1270, 1236, 593, 862, 1100, 476, 468, 777, 734, 295, 142, 721, 1108, 51, 83, 210, 697, 1035, 933, 384, 850, 555, 525, 1198, 1278, 257, 848, 687, 1133, 105, 289, 50, 283, 914, 656, 588, 1090, 764, 67, 383, 931, 1287, 612, 881, 1259, 894, 874, 1143, 1206, 619, 979, 953, 1163, 354, 1011, 1056, 1303, 843, 463, 505, 746, 164, 26, 1066, 609, 511, 629, 1169, 1164, 288, 940, 136, 1247, 801, 769, 146, 93, 633, 217, 216, 1077, 608, 462, 738, 1153, 1293, 791, 826, 982, 663, 1120, 944, 564, 1292, 1008, 1172, 1266, 294, 530, 299, 650, 224, 1282, 727, 499, 1200, 495, 13, 658, 583, 788, 1045, 867, 152, 1243, 80, 757, 1109, 296, 1089, 79, 393, 507, 473, 1123, 792, 242, 856, 1301, 260, 487, 68, 1168, 1255, 550, 1028, 72, 469, 725, 700, 503, 719, 998, 644, 875, 672, 1181, 573, 464, 387, 858, 371, 1161, 137, 846, 32, 88, 1291, 740, 600, 17, 1208, 1232, 996, 143, 778, 1302, 1224, 527, 807, 1309, 904, 981, 292, 1062, 766, 1204, 1031, 1136, 676, 906, 543, 601, 741, 240, 647, 1231, 308, 24, 405, 43, 1022, 1086, 148, 835, 1194, 94, 6, 1128, 964, 207, 674, 516, 834, 774, 832, 739, 345, 2, 830, 880, 225, 318, 36, 252, 333, 562, 338, 374, 271, 1063, 790, 827, 163, 191, 472, 1205, 1105, 1002, 250, 1249, 290, 689, 763, 1147, 684, 1241, 1044, 1245, 886, 263, 643, 73, 15, 808, 1104, 1023, 416, 1297, 654, 1073, 887, 205, 352, 237, 222, 831, 317, 520, 1067, 561, 838, 675, 444, 668, 677, 968, 1289, 1124, 488, 528, 196, 566, 947, 753, 784, 16, 38, 258, 869, 357, 279, 184, 691, 514, 1091, 298, 1034, 780, 973, 314, 1254, 1038, 208, 852, 52, 1037, 888, 113, 246, 729, 541, 45, 1122, 639, 276, 1145, 568, 883, 12, 1069, 921, 272, 69, 229, 625, 86, 268, 1088, 91, 652, 18, 277, 913, 1027, 1150, 1110, 986, 965, 1093, 92, 336, 1039, 508, 586, 885]
    M_ids_val = [551, 154, 417, 1196, 841, 695, 1030, 842, 133, 776, 278, 1295, 800, 41, 280, 273, 87, 1140, 236, 1152, 116, 66, 945, 304, 1118, 1230, 341, 395, 19, 723, 1071, 1010, 569, 348, 1119, 1001, 742, 95, 328, 1055, 202, 433, 349, 415, 529, 443, 726, 1219, 231, 855, 686, 168, 500, 409, 97, 767, 226, 480, 923, 261, 147, 376, 679, 661, 736, 1227, 969, 126, 805, 312, 688, 219, 927, 440, 1024, 690, 75, 325, 1281, 403, 1212, 428, 1217, 460, 863, 1135, 917, 1080, 1192, 1251, 900, 537, 1074, 1290, 1121, 491, 829, 548, 522, 1042, 962, 712, 206, 526, 623, 519, 630, 976, 455, 65, 1218, 1075, 781, 29, 577, 465, 521, 101, 1207, 1036, 285, 197, 256, 59, 575, 907, 109, 991, 1155, 496, 570, 545, 1078, 701, 1220, 64, 1186, 124, 745, 636, 486, 1294, 1300, 155, 821, 797, 5, 626, 189, 1058, 452, 896, 20, 524, 538, 157, 546, 1213, 1025, 902, 943, 849, 432, 531, 928, 837, 599, 681, 471, 1189, 364, 760, 935, 63, 135, 857, 431, 559, 267, 335, 603, 785, 248, 515, 1041, 437, 517, 1081, 434, 400, 1054, 287, 212, 30, 44, 270, 321, 1222, 1262, 301, 1253, 1137, 57, 262, 350, 327, 1060, 1188, 952] 
    M_ids_test = [813, 179, 683, 1299, 367, 983, 410, 798, 247, 175, 707, 985, 4, 992, 420, 694, 404, 966, 1087, 1115, 453, 166, 749, 718, 918, 708, 398, 532, 971, 574, 58, 60, 1096, 1013, 1064, 411, 466, 1156, 492, 401, 1095, 678, 1033, 1094, 458, 731, 344, 754, 1065, 536, 188, 613, 1107, 56, 302, 1250, 1085, 422, 48, 1061, 332, 984, 544, 704, 1139, 1275, 177, 641, 994, 987, 958, 1003, 238, 253, 975, 748, 714, 518, 306, 773, 427, 145, 53, 747, 209, 1102, 1174, 1, 211, 108, 1223, 227, 334, 905, 429, 138, 392, 128, 823, 990, 214, 810, 820, 818, 806, 1149, 939, 111, 1059, 259, 655, 759, 1098, 1276, 300, 9, 579, 540, 375, 997, 25, 479, 477, 1225, 362, 171, 311, 474, 782, 737, 853, 572, 1130, 884, 282, 571, 1129, 825, 1239, 342, 167, 1257, 803, 817, 509, 266, 590, 448, 121, 1015, 241, 346, 730, 949, 47, 713, 793, 584, 915, 1187, 637, 125, 134, 974, 554, 193, 141, 938, 1191, 1021, 1221, 720, 699, 645, 1177, 1146, 160, 81, 866, 1005, 313, 703, 339, 1298, 1052, 1273, 151, 591, 922, 706, 1017, 1079, 617, 1180, 169, 501, 370, 510, 616, 743, 932, 1284, 957, 646, 878, 1092, 1020, 423, 123, 1159, 129, 1125, 201, 765, 386, 1272, 1195, 213, 372, 1151, 634, 1173, 243, 1111, 297, 861, 595, 274, 1144, 96, 454, 102, 815, 1203, 1185, 127, 1305, 1288, 459, 255, 1162, 122, 872, 1132, 187, 494, 144, 859, 891, 178, 77, 441, 1043, 119, 1197, 751, 194, 1101, 150, 99, 811, 310, 666]
    return M_ids_train, M_ids_val, M_ids_test
#%% google get dfs
def get_df_google():
    from args_google import get_paths
    base_path, _, _, _, feat_step3, sav_path, _ = get_paths()
    target = 'cpu_utilization'
    id_m = "machine_id"
    sort_by = 'start_time'

    df = loadDatasetObj(os.path.join(base_path, 'google.obj'))

    # dat_obi = os.path.join(feat_step3,'XY_test_ready.obj')
    selected_machines = [104792148941, 105123822036, 10880083972, 1377327200, 143873943769, 151931027, 1579670885, 1713162907, 1715744191, 19882051227, 21305998, 21372228, 21399559, 22506461, 22996256, 23749075262, 23749170581, 23749187178, 23859188838, 24270907240, 249752968639, 25536824701, 3093041819, 32972586955, 330553462941, 334582773765, 346015909090, 348407234491, 350002422714, 35970888485, 373220204176, 373578788446, 375532102846, 4450793427, 4451038352, 4451164031, 4451430235, 4468823952, 62155170, 70551104365, 70599997294, 70609032639, 70609462184, 71979925955, 92005174092, 92029884620, 92039127074, 92055701590, 92062361748, 92147238400, 96914881882, 96914917362, 96921743123, 96921783666, 96936815458]
    #list(loadDatasetObj(dat_obi)['XY_test_ready']['M_id'].unique())
    df = df[df[id_m].isin(selected_machines)]

    df.dropna(inplace=True)
    normalize_cols = ['y', 'x1']
    rename_map = {
        target: "y",
        'memory_utilization': "x1",
        id_m: "id",
        sort_by:'date',
    }
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df = df.loc[:,list(rename_map.values())]
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler
    #2020-04-01,
    offset_seconds = 600
    df['date'] = pd.to_datetime(df['date'] / 1e6 + offset_seconds, unit='s')
    start_date = pd.to_datetime('2019-05-01')
    df['date'] = start_date + (df['date'] - pd.to_datetime('1970-01-01'))

    df = df.sort_values(['id', 'date'])
    

    df.reset_index(drop=True, inplace=True)
    
    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
        df,train_val_per ,val_per)

    df_train = df_from_M_id(df, M_ids_train)
    df_val = df_from_M_id(df, M_ids_val)
    df_test = df_from_M_id(df, M_ids_test)

    return df_train, df_val, df_test
#%% bitbrain get df
def get_df_BB():
    
    from args_BB import get_paths
    base_path, processed_path, feat_BB_step1, feat_BB_step2, feat_step3, sav_path, sav_path_plot = get_paths()
    
    target = 'CPU usage [%]'
    id_m = "machine_id"
    sort_by = 'date'
    # -----------------------
    # --- Load Data ---

    df = loadDatasetObj(os.path.join(base_path, 'rnd.obj'))
    df['memory_utilization'] = (df['Memory usage [KB]'] / df['Memory capacity provisioned [KB]']) * 100
    df.loc[df['Memory capacity provisioned [KB]'] == 0, 'memory_utilization'] = 0
 
    df['memory_utilization'] = df['memory_utilization'].clip(lower=0, upper=100)
  
    df.dropna(inplace=True)
    normalize_cols = ['y', 'x1']
    rename_map = {
        target: "y",
        'memory_utilization': "x1",
        id_m: "id",
        sort_by:'date',
    }
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df = df.loc[:,list(rename_map.values())]
    
    k_threshold = 10
    # --- Filter Machines based on Average CPU Usage ---
    average_values = df.groupby("id")['y'].mean()
    selected_machines = average_values[average_values > k_threshold].index.tolist()
    df = df[df["id"].isin(selected_machines)]

    
    
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler

    # reference_time = datetime(2013, 8, 1)
    # df['date'] = pd.to_datetime(df['date'], unit='ms')

    
    df = df.sort_values(['id', 'date'])
    

    df.reset_index(drop=True, inplace=True)
    
    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
        df,train_val_per ,val_per)

    df_train = df_from_M_id(df, M_ids_train)
    df_val = df_from_M_id(df, M_ids_val)
    df_test = df_from_M_id(df, M_ids_test)

    return df_train, df_val, df_test



#%% main function to use get_dicts
def get_dict_df(df,seq_length):
    X_list,Y_list,M_ids = window_rolling(df,seq_length)
    dict_df = {'X_list':X_list,'Y_list':Y_list,'M_ids':M_ids,
               'X':list_to_array(X_list),'Y':list_to_array(Y_list)}
    return dict_df

def get_dicts(df_train, df_val, df_test,seq_length):
    train_dict = get_dict_df(df_train,seq_length)
    val_dict = get_dict_df(df_val,seq_length)
    test_dict = get_dict_df(df_test,seq_length)
    return train_dict , val_dict, test_dict

def get_dict_option(flag,seq_length):
    if flag == "Alibaba":
        df_train, df_val, df_test = get_df_alibaba()
    elif flag == "google":
        df_train, df_val, df_test = get_df_google()
    elif flag == "BB":
        df_train, df_val, df_test = get_df_BB()
        
    train_dict , val_dict, test_dict = get_dicts(df_train, df_val, 
                                                 df_test,seq_length)
    return train_dict , val_dict, test_dict

        

#%%
# def model_serv(X_list,Y_list,model,scaler,batch_size):
#     y_test_pred_list = []
#     rmse_list = []
#     for c,test_sample_all in enumerate(X_list):
#         if len(test_sample_all)>batch_size:
#             test_sample_all = split_3d_array(test_sample_all, batch_size)
#         else:
#             test_sample_all = [test_sample_all]
#         pred_i = []
#         for test_sample in test_sample_all:
#             pred_ii = list(np.squeeze(np.array(model.predict(test_sample))) *scaler)
#             pred_i.append(pred_ii)
#         pred_i = flatten(pred_i)
#         y_test_pred_list.append(pred_i)
#         rmse_i_list = RMSE(Y_list[c]*scaler,pred_i)
#         rmse_list.append(rmse_i_list)
#     return y_test_pred_list,rmse_list

def model_serv(X_list,Y_list,model,scaler,batch_size):
    y_test_pred_list = []
    rmse_list = []
    for c,test_sample in enumerate(X_list):
        pred_i = model.predict(test_sample) *scaler
        y_test_pred_list.append(pred_i )

        rmse_i_list = RMSE(Y_list[c]*scaler,pred_i)
        rmse_list.append(rmse_i_list)
    return y_test_pred_list,rmse_list

#%%
def log_results_EB0(row,cols,save_name):
    if not os.path.isfile(save_name):
        df3 = pd.DataFrame(columns=cols)
        df3.to_csv(save_name,index=False)   
    df = pd.read_csv(save_name)
    df.loc[len(df)] = row
    flag = 0
    if len(df)!=0:
        if row[0] == df.min()['RMSE']:
            flag = 1
    else:
        flag = 1
    df.to_csv(save_name,mode='w', index=False,header=True)
    return flag

#%%
def separate_XY(df,id_m):
    X = np.array(df.drop([id_m,'y'],axis=1))
    y  = np.array(df['y'])
    return X,y
def get_from_id(df,M,id_m):
    return df.loc[df[id_m].isin(M)]
def get_adaptive_data(flag_dataset):
    id_m = 'M_id'
    if flag_dataset == 'google':
        from args_google import get_paths
        base_path,processed_path,feat_google_step1,feat_google_step2,feat_step3,sav_path,sav_path_plots = get_paths()
        df_train, df_val, df_test = get_df_google()
    elif flag_dataset == 'BB':
        from args_BB import get_paths
        base_path, processed_path, feat_BB_step1, feat_BB_step2, feat_step3, sav_path, sav_path_plot = get_paths()
        df_train, df_val, df_test = get_df_BB()
    elif flag_dataset == 'Alibaba':
        from args import get_paths
        base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_step3,sav_path,sav_path_plots = get_paths()
        dat_path_obi = os.path.join(feat_step3,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
        X_train,y_train,X_test,y_test,scaler,df_test_xy = get_data_stat(dat_path_obi)
        return X_train,y_train,X_test,y_test,scaler,df_test_xy

    train_ids, val_ids, test_ids = list(df_train['id']),list(df_val['id']),list(df_test['id'])
    del df_train, df_val, df_test
    dat_obi = os.path.join(feat_step3,'XY_test_ready.obj')
    df_test_xy = loadDatasetObj(dat_obi)['XY_test_ready']
    
    

    X_train,y_train =  separate_XY(get_from_id(df_test_xy, train_ids,id_m),id_m)
    df_test_XY = get_from_id(df_test_xy, val_ids+test_ids,id_m)
    X_test,y_test =  separate_XY(df_test_XY,id_m)
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train,y_train,X_test,y_test,scaler,df_test_XY
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def reg_all(X_train,y_train,X_test,reg_model):
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
    from sklearn.svm import SVR,LinearSVR
    from sklearn.linear_model import LinearRegression
    # LinearSVR(max_iter=100000) 
    # SVR(kernel= 'linear', cache_size=4000, n_jobs=-1)
    reg_models_names = ["linear_reg","svr_reg","GPR_reg","GBT_reg"]
    ind = [c for c,ele in enumerate(reg_models_names) if ele==reg_model][0]
    regs_all = [LinearRegression(), LinearSVR(max_iter=100000) ,GaussianProcessRegressor(copy_X_train=False,kernel=DotProduct() + WhiteKernel()), 
                HistGradientBoostingRegressor()]
    reg = regs_all[ind]
    reg.fit(X_train, y_train)
    
    return reg#,reg.predict(X_train),reg.predict(X_test)





def class_all(X_train,y_train,X_test,class_model):
    from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier,HistGradientBoostingClassifier
    class_models_names = ["KNN","MLP","GNB","RDF","GBT"]
    ind = [c for c,ele in enumerate(class_models_names) if ele==class_model][0]
    class_all = [KNeighborsClassifier(n_neighbors=3), "keras",
                GaussianNB(),RandomForestClassifier(n_estimators=50,max_depth=12),
                HistGradientBoostingClassifier()]
    classifier = class_all[ind]
    if class_model == "MLP":
        classifier,y_test_pred = MLP_classifier(X_train,y_train,X_test)
    else: 
        classifier.fit(X_train, y_train)
        y_test_pred = classifier.predict(X_test)
    
    return classifier,y_test_pred








def MLP_classifier(X_train,y_train,X_test):
    from keras.layers import Dense
    from keras.models import  Sequential #,load_model
    
    import os
    
    class SaveBestModel(tf.keras.callbacks.Callback):
        def __init__(self, save_best_metric='val_loss', this_max=False):
            self.save_best_metric = save_best_metric
            self.max = this_max
            if this_max:
                self.best = float('-inf')
            else:
                self.best = float('inf')
    
        def on_epoch_end(self, epoch, logs=None):
            metric_value = logs[self.save_best_metric]
            if self.max:
                if metric_value > self.best:
                    self.best = metric_value
                    self.best_weights = self.model.get_weights()
    
            else:
                if metric_value < self.best:
                    self.best = metric_value
                    self.best_weights= self.model.get_weights()
    
    #%%
    out_size = len(np.unique(y_train))
    y_train = tf.keras.utils.to_categorical(y_train)
    n = 2**8

    model = Sequential()
    model.add(Dense(n, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(n, activation='relu'))
    # model.add(Dense(n, activation='relu'))

    model.add(Dense(out_size, activation='softmax'))

    checkpoint = SaveBestModel()
    callbacks_list = [checkpoint]
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=150, batch_size=2048, validation_split =0.2,callbacks=callbacks_list)
    model.set_weights(checkpoint.best_weights)
    # best_epoch = np.argmin(history.history['val_loss'])
    
    y_test_pred = np.argmax(model.predict(X_test),axis=1)
    return model,y_test_pred

#%%



def conv_block(x, filters, kernel_size, strides=1, activation='swish', use_bn=True):
    from tensorflow.keras import layers
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def attention_block(x, heads=4, hidden_dim=128):
    from tensorflow.keras import layers
    _, seq_len, _ = x.shape
    
    # Multi-head attention
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=hidden_dim // heads)(x, x)
    
    # Add & Norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x

def simplified_inverted_residual_block(x, filters, kernel_size=3, strides=1, activation='swish', use_bn=True, se_ratio=None):
    # Depthwise convolution
    from tensorflow.keras import layers
    x_dw = layers.DepthwiseConv1D(kernel_size, strides=strides, padding='same', use_bias=not use_bn)(x)
    if use_bn:
        x_dw = layers.BatchNormalization()(x_dw)
    x_dw = layers.Activation(activation)(x_dw)

    # Squeeze and excite (if enabled)
    if se_ratio is not None:
        x_dw = squeeze_excite_block(x_dw, se_ratio)

    # Projection phase (1x1 convolution)
    x_project = layers.Conv1D(filters, 1, padding='same', use_bias=not use_bn)(x_dw)
    if use_bn:
        x_project = layers.BatchNormalization()(x_project)

    # Residual connection (only if strides are 1 and input/output channels are the same)
    if strides == 1 and x.shape[-1] == filters:
        # Concatenate the input and the projected output
        x = layers.Concatenate()([x, x_project]) # Changed to concatenation
    else:
        x = x_project
    return x

def squeeze_excite_block(x, se_ratio=0.25):
    from tensorflow.keras import layers
    channels = x.shape[-1]
    se_shape = (1, channels)

    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(int(channels * se_ratio), activation='swish')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    x = layers.Multiply()([x, se])
    return x

def Enhanced_EfficientNet_Like_1D(input_shape, num_classes=1, base_filters=32, blocks_per_stage=[1, 2, 2, 1],
                                    kss=[3, 5, 3], activation='swish', use_se=True, use_bn=True):
    from tensorflow.keras import layers
    from tensorflow import keras
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Initial convolution
    x = conv_block(x, base_filters, kernel_size=3, strides=1, activation=activation, use_bn=use_bn)

    # Inverted residual blocks with attention
    for i, num_blocks in enumerate(blocks_per_stage):
        num_filters = base_filters * (2 ** i)
        for j in range(num_blocks):
            x = simplified_inverted_residual_block(x, num_filters, kernel_size=kss[i],
                                                    strides=2 if j == 0 and i < 3 else 1,
                                                    activation=activation,
                                                    use_bn=use_bn,
                                                    se_ratio=0.25 if use_se else None)
            x = layers.Activation(activation)(x)  # Activation after residual block
        x = attention_block(x, heads=4, hidden_dim=num_filters)
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dropout (add back in if needed)
    x = layers.Dropout(0.4)(x)  # Example dropout rate

    # Output layer
    outputs = layers.Dense(num_classes, activation='linear')(x)

    model = keras.Model(inputs, outputs)
    return model 
    

#%%


from tensorflow import keras
from tensorflow.keras import layers

def get_CEDL(encoder_inputs,units):
    num_layers = 1
    encoder_outputs = layers.LSTM(units, return_sequences=True)(encoder_inputs)
    for i in range(num_layers - 1):
        encoder_outputs = layers.LSTM(units, return_sequences=True, name=f"encoder_lstm_{i+1}")(encoder_outputs)
    encoder_outputs, state_h, state_c = layers.LSTM(units, return_sequences=True, return_state=True, name=f"encoder_lstm_{num_layers}")(encoder_outputs)
    encoder_states = [state_h, state_c]
    context = layers.RepeatVector(1, name="context_vector")(encoder_outputs[:, -1, :])
    decoder = layers.LSTM(units, return_sequences=True, name="decoder_lstm_0")(context, initial_state=encoder_states)
    for i in range(num_layers - 1):
        decoder = layers.LSTM(units, return_sequences=True, name=f"decoder_lstm_{i+1}")(decoder)
    attention_layer = layers.AdditiveAttention(name="attention_layer")
    attention_output = attention_layer([decoder, encoder_outputs])
    decoder_combined_context = layers.concatenate([decoder, attention_output], axis=-1)
    outputs = layers.Dense(units, activation='swish')(decoder_combined_context)
    outputs = layers.Flatten()(outputs)
    return outputs

def create_patch_tst_lstm_hybrid(input_shape, pred_len=1, 
                 patch_length=16, num_heads=4,LSTM_units=256):

    inputs = keras.Input(shape=input_shape)  # Input shape: (sequence_length, num_features)
    sequence_length = input_shape[0]
    num_features = input_shape[1]

    # 1. Patch TST Block
    # Divide the time series into patches
    num_patches = sequence_length // patch_length
    x_patches = layers.Reshape((num_patches, patch_length * num_features))(inputs)  # Shape: (batch_size, num_patches, patch_length * num_features)
    
    # Transformer Encoder for Patch TST
    transformer_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=patch_length * num_features)(x_patches, x_patches)
    transformer_output = layers.LayerNormalization(epsilon=1e-6)(transformer_output + x_patches)  # Add & Norm
    transformer_output = layers.GlobalAveragePooling1D()(transformer_output)  # Aggregate patch-level features

    # 2. LSTM Block
    x_lstm = get_CEDL(inputs,LSTM_units)
    x = layers.concatenate([transformer_output, x_lstm])  # Combine Patch TST and LSTM outputs

    # 4. Dense Layers

    x = layers.Dense(LSTM_units*2, activation="swish")(x)
    outputs = layers.Dense(pred_len)(x)  # Predict the next `pred_len` time steps
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Huber(delta=1) 'MAE'
    model.compile(optimizer='adam', loss='MAE',metrics=[ 'mape']) 
    return model
#%%

def get_best_lsmt_para(flag,flag_dataset=0):
    #flag 0 for En-De LSMT, 1 for normal lstm
    if flag == 0:
        return {'units': 256, 'num_layers': 1, 'seq': 29, 'dense_units': 256}
    elif flag == 1:
        return {'units': 256, 'num_layers': 2, 'seq': 20, 'dense_units': 256}
    elif flag == 2:
        if flag_dataset == 0:
            return   {'LSTM_units': 256,'num_heads': 4}
        else:
            return   {'LSTM_units': 256,'num_heads': 4}
        #%%
def get_hybrid_hyperparameters(x):
    """Get hyperparameters for solution `x`."""
    if isinstance(x, list) and x==[]:
        x = [0,0]
    LSTM_units = 2**int(x[0]*4 + 6)
    num_heads = int(x[1]*8+2)
    params =  {
        'LSTM_units': LSTM_units,
        'num_heads': num_heads,
    }
    # print(params)
    return params

def get_hyperparameters_LSTM(x):
    if x == []:
        x = [0,0,0,0]
    """Get hyperparameters for solution `x`."""
    units = 2**int(x[0]*4 + 6)
    num_layers = int(x[1]*6+1)
    seq = int(x[2]*31 + 1)
    dense_units = 2**int(x[3]*3 + 7)
    params =  {
        'units': units,
        'num_layers': num_layers,
        'seq':seq,
        'dense_units':dense_units
    }
    # print(params)
    return params


def switch_para_CSA(model_name):
    if model_name == 'LSTM':
        func_para = get_hyperparameters_LSTM
    elif model_name == 'EnDeAtt':
        func_para = get_hyperparameters_LSTM
    elif model_name == 'TST_LSTM': 
        func_para = get_hybrid_hyperparameters
    return func_para
        
        
def switch_model_CSA(model_name,input_dim,output_dim,params):
    if model_name == 'LSTM':
        model = get_lstm_model(input_dim,output_dim,**params)
    elif model_name == 'EnDeAtt':
        model = get_en_de_lstm_model_attention(input_dim,output_dim,**params)
    elif model_name == 'TST_LSTM':
        # model = create_patch_tst_lstm_hybrid(input_dim,**params)
        model = temposightV2(input_dim,**params)
    return model
      

def get_data_CSA(params,data_set):
    scaler=100
    if 'seq' not in list(params.keys()):
        seq = 32
    else:
        seq = params['seq']
    train_dict , val_dict, test_dict = get_dict_option(data_set,seq)

    X_train = train_dict['X']
    y_train = train_dict['Y']
    
    X_val = val_dict['X']
    y_val = val_dict['Y']
    
    X_test_list = test_dict['X_list']
    y_test_list = test_dict['Y_list']
    
    X_test = test_dict['X']
    y_test = test_dict['Y']
    
    Mids_test = test_dict['M_ids']
    
    y_train = expand_dims(expand_dims(y_train))
    y_val = expand_dims(expand_dims(y_val))
    return X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test




#%%


def transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    """
    A full Transformer block consisting of:
    - MultiHeadAttention
    - Add & Norm (LayerNormalization + Residual Connection)
    - Feed-Forward Network (FFN)
    - Add & Norm
    """
    # MultiHeadAttention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    # Add & Norm
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-Forward Network (FFN)
    ffn_output = layers.Dense(ff_dim, activation="relu")(x)
    ffn_output = layers.Dense(key_dim)(ffn_output)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    # Add & Norm
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    return x

def temposightV2(input_shape, pred_len=1, patch_length=16, num_heads=4, LSTM_units=256):
    inputs = layers.Input(shape=input_shape)  # Input shape: (sequence_length, num_features)
    sequence_length = input_shape[0]
    num_features = input_shape[1]

    # 1. Patch TST Block
    # Divide the time series into patches
    num_patches = sequence_length // patch_length
    x_patches = layers.Reshape((num_patches, patch_length * num_features))(inputs)  # Shape: (batch_size, num_patches, patch_length * num_features)

    # Add two full Transformer layers
    for _ in range(2):  # Two Transformer layers
        x_patches = transformer_block(x_patches, num_heads=num_heads, key_dim=patch_length * num_features, ff_dim=512)

    # Aggregate patch-level features
    transformer_output = layers.GlobalAveragePooling1D()(x_patches)

    # 2. LSTM Block
    x_lstm = get_CEDL(inputs, LSTM_units)

    # 3. Combine Patch TST and LSTM outputs
    x = layers.concatenate([transformer_output, x_lstm])

    # 4. Dense Layers
    x = layers.Dense(LSTM_units * 2, activation="swish")(x)
    outputs = layers.Dense(pred_len)(x)  # Predict the next `pred_len` time steps

    # Build the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='MAE', metrics=['mape'])
    return model