import numpy as np
import matplotlib.pyplot as plt
import time
import hdbscan

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = '../artificial/'
name0 = "xclara.arff"
name1 = "2d-4c.arff" #simple
name2 = "diamond9.arff"#simple
name3 = "banana.arff" #difficile
name4 = "complex8.arff" #difficile
name5 = "birch-rg3.arff" #grand dataset
name = name4

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Distances aux k plus proches voisins
# Donnees dans X
k = 5
neigh = NearestNeighbors(n_neighbors = k)
neigh.fit(datanp)
distances, indices = neigh.kneighbors(datanp)
# distance moyenne sur les k plus proches voisins
# en retirant le point " origine "
newDistances = np.asarray([np.average(distances[i] [1:]) for i in range(0, distances.shape[0])])
# trier par ordre croissant
distancetrie = np.sort(newDistances)
plt.title("Plus proches voisins " + str(k))
plt.plot(distancetrie)
plt.show()


# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
tps1 = time.time()
epsilon= 10 #2  # 4
min_pts= k #10   # 10
min_cluster_size = 55
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
#model = hdbscan.HDBSCAN(min_samples = min_pts, min_cluster_size = min_cluster_size)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Epsilon: ',epsilon)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)
print('Runtime: ' , round((tps2 - tps1)*1000,2),"ms")

plt.scatter(f0, f1, c=labels, s=8)
# plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.title("DBSCAN (1) - min_pts= "+str(min_pts)+" epsilon= "+str(epsilon))
plt.show()


####################################################
# Standardisation des donnees

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()


print("------------------------------------------------------")
print("Appel HDBSCAN (2) sur données standardisees ... ")
tps1 = time.time()
epsilon=0.05 #0.05
min_pts=5 # 10
min_cluster_size = 20
#model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model = hdbscan.HDBSCAN(min_samples = min_pts, min_cluster_size = min_cluster_size)
model.fit(data_scaled)

tps2 = time.time()
labels = model.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
#plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.title("HDBSCAN (2) - min_samples= "+str(min_pts)+" min_cluster_size= "+str(min_cluster_size))
plt.show()







