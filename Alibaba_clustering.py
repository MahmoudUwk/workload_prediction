import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import pickle
import os

from Alibaba_helper_functions import loadDatasetObj,save_object
        
        
def save_M_ids(M_id_labels,df,M_ids,sav_path):
    indeces_Mid = [M_ids[np.where(M_id_labels==class_Mid)[0]] for  class_Mid in np.unique(M_id_labels)]
    save_object(indeces_Mid, sav_path)
    
def plot_box(M_id_labels,df,des,M_ids):
    indeces_Mid = [M_ids[np.where(M_id_labels==class_Mid)[0]] for  class_Mid in np.unique(M_id_labels)]
    print([len(inds) for inds in indeces_Mid])
    # dfs = [df_from_M_id(df,indeces_Mid[0]) for ind in range(kmeans.n_clusters)]
    df_list = []
    len_dfs = []
    for ind_m_id in indeces_Mid:
        dummy = df_from_M_id(df,ind_m_id)
        len_dfs.append(len(dummy))
        df_list.append(dummy)
    # df_M1 = df_from_M_id(df,indeces_Mid[0])
    
    # df_M2 = df_from_M_id(df,indeces_Mid[1])
    print(len(df)-sum(len_dfs))
    
    labels = ["M"+str(ind+1) for ind in range(len(np.unique(M_id_labels)))]
    
    plt.figure()
        
    plt.boxplot(df_list, labels=labels)
    plt.title('Clutering using'+des)

def df_from_M_id(df,M):
    return df.loc[df[" machine id"].isin(M)][" used percent of cpus(%)"]

sav_path = "data/features_lstm/"
script = "server_usage.csv"
target = " used percent of cpus(%)"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

base_path = "data/"
# base_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/"
info_path = base_path+"schema.csv"

df_info =  pd.read_csv(info_path)
df_info = df_info[df_info["file name"] == script]['content']



full_path = base_path+script
nrows = None
df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))

df = df[[" machine id", " timestamp"," used percent of cpus(%)"]]
# df = df[df.notna()]
df = df.dropna()

df_grouped_id = df.groupby([" machine id"])
#%%
len_all = []
for M_id, M_id_val in df_grouped_id:
    len_all.append(len(M_id_val))
min_len = min(len_all)

X = np.zeros((len(df_grouped_id),min_len))
M_ids=np.zeros((len(df_grouped_id)))
c_i = 0
for  M_id, M_id_val in df_grouped_id:
    M_id_val = np.array(M_id_val.sort_values(by=[' timestamp']).reset_index(drop=True).drop([" machine id"," timestamp"],axis=1))[:min_len].copy()
    X[c_i,:] = np.squeeze(M_id_val)
    M_ids[c_i] = M_id[0]
    c_i+=1

#%%
from sklearn.cluster import AgglomerativeClustering

distance = 'l1'
clustering = AgglomerativeClustering(n_clusters=4,affinity=distance,linkage='single').fit(X)

des = 'AgglomerativeClustering using '+distance
plot_box(clustering.labels_,df,des,M_ids)

sav_path_TimeSeriesKMeans = os.path.join(sav_path,'TimeSeriesKMeans.obj')
save_M_ids(clustering.labels_,df,M_ids,sav_path_TimeSeriesKMeans)
#%%

from tslearn.clustering import TimeSeriesKMeans

model = TimeSeriesKMeans(n_clusters=4, metric="softdtw",max_iter=20)
model.fit(X)
#%%
des = 'TimeSeriesKMeans using Dynamic Time Warping'
plot_box(model.labels_,df,des,M_ids)

sav_path_TimeSeriesKMeans = os.path.join(sav_path,'TimeSeriesKMeans'+str(model.n_clusters)+'.obj')
save_M_ids(model.labels_,df,M_ids,sav_path_TimeSeriesKMeans)
#%%

# [M_ids[ for label in model.labels_]
'''
mean_Mid= np.expand_dims(np.array(df_grouped_id.mean()[target]), axis=1)
std_Mid= np.expand_dims(np.array(df_grouped_id.std()[target]), axis=1)
X_Mid = np.concatenate((mean_Mid,std_Mid),axis=1)



# kmeans = KMeans(n_clusters=4, n_init="auto",max_iter=5000,random_state=5).fit(X_Mid)
kmeans = KMeans(n_clusters=4,algorithm='elkan',max_iter=5000,random_state=5).fit(X_Mid)

M_id_labels = kmeans.labels_

indeces_Mid= [np.where(M_id_labels==class_Mid)[0] for  class_Mid in np.unique(kmeans.labels_)]

# dfs = [df_from_M_id(df,indeces_Mid[0]) for ind in range(kmeans.n_clusters)]

df_M1 = df_from_M_id(df,indeces_Mid[0])

df_M2 = df_from_M_id(df,indeces_Mid[1])

df_M3 = df_from_M_id(df,indeces_Mid[2])

df_M4 = df_from_M_id(df,indeces_Mid[3])

# print(len(df_M1),len(df_M2),len(df_M3),len(df_M4))

plt.figure()
ax = plt.boxplot([df_M1,df_M2,df_M3,df_M4], labels=["M1","M2","M3","M4"])

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X_Mid[:, 0].min() - 1, X_Mid[:, 0].max() + 1
y_min, y_max = X_Mid[:, 1].min() - 1, X_Mid[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1,figsize=(10,7),dpi=160)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(X_Mid[:, 0], X_Mid[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on Mean and Std"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.xlabel("Mean CPU Utilization")
plt.ylabel("Std CPU Utilization")
plt.yticks(())
plt.show()
'''

    #%%
# import seaborn as sns
# df_corr = df[cols].corr()
# plt.figure(figsize=(10,7),dpi=180)
# xticks_font_size = 5
# # sns.set(font_scale=1.2)
# plt.rc('xtick', labelsize=xticks_font_size)
# plt.rc('ytick', labelsize=xticks_font_size)
# sns.heatmap(df_corr.round(2), annot=True, annot_kws={"size": 10})
# plt.savefig(os.path.join(sav_path,"corr.png"))










