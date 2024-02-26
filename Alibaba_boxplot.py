# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:23:49 2024

@author: mahmo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:09:55 2024

@author: mahmo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def df_from_M_id(df,M):
    return df.loc[df[" machine id"].isin(M)][" used percent of cpus(%)"]

script = "server_usage.csv"
target = " used percent of cpus(%)"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

base_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/"
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

mean_Mid= np.expand_dims(np.array(df_grouped_id.mean()[target]), axis=1)
std_Mid= np.expand_dims(np.array(df_grouped_id.std()[target]), axis=1)
X_Mid = np.concatenate((mean_Mid,std_Mid),axis=1)

#%%



kmeans = KMeans(n_clusters=4, n_init="auto",algorithm='elkan',max_iter=5000,random_state=5).fit(X_Mid)

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
#%%

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
#%%

# df_grouped_id.mean()[target].hist()

# mean_therhold = [5,30]
# values_thershold = 63

# M1 = np.where(df_grouped_id.mean()[target]>mean_therhold[1])[0]
# M2 = np.where(df_grouped_id.mean()[target]<mean_therhold[0])[0]

# df_M1 = df_from_M_id(df,M1)
# df_M2 = df_from_M_id(df,M2)



# M34 = np.array(list(set(list(df[" machine id"])) - set(list(M1)+list(M2))))


# df_34 = df.loc[df[" machine id"].isin(M34)]

# df_grouped_id = df_34.groupby([" machine id"])

# df_grouped_id.std()[target].hist()
# std_therhold = 1

# M4 = np.where(df_grouped_id.std()[target]<std_therhold)[0]
# M3 = np.where(df_grouped_id.std()[target]>std_therhold)[0]


# df_M4 = df_from_M_id(df_34,M4)
# df_M3 = df_from_M_id(df_34,M3)



# print(len(df_M1),len(df_M2),len(df_M3),len(df_M4))
# import matplotlib.pyplot as plt
# plt.figure()
# ax = plt.boxplot([df_M1,df_M2,df_M3,df_M4], labels=["M1","M2","M3","M4"])













