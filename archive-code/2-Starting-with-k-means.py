"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering


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

#Calcul silhouette
from sklearn.metrics import silhouette_score
sils = []
def calcul_silhouette():
    for k in range (2,20):
        _model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
        _model.fit(datanp)
        labelsil = _model.labels_
        sil = silhouette_score(datanp, labelsil)
        sils.append(sil)
    plt.scatter(range(2,20), sils, s=6)
    plt.plot(range(2,20), sils)
    plt.xlabel("Nombre de clusters(k)")
    plt.ylabel("Coefficient de silhouette")
    plt.show()
start_time = time.time()
calcul_silhouette()
k = np.argmax(sils)+2
end_time = time.time()
total_time = end_time - start_time
print("total time = ", total_time)

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
#k=4
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)


######################################################################





def calcul_inertie() :
    inerties = []
    for k in range(1,51):
        _model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
        _model.fit(datanp)
        inertie = _model.inertia_
        inerties.append(inertie)
    plt.scatter(range(1,51), inerties, s=6)
    plt.plot(range(1,51), inerties)
    plt.xlabel("Nombre de clusters(k)")
    plt.ylabel("Inertie")
    plt.show()

def regroupement():
    # Afficher les scores de regroupement pour chaque point de données
    for i in range(k):
        distances = []
        for n in datanp[labels== i] :
            distances.append(euclidean_distances([n, centroids[i]])[0][1])
        score_min = min(distances)  # Distance minimale dans le cluster
        score_max = max(distances)  # Distance maximale dans le cluster
        score_avg = sum(distances)/len(distances)  # Distance moyenne dans le cluster
        print(f"Cluster {i + 1} - Score de regroupement (Distance minimale) : {score_min}")
        print(f"Cluster {i + 1} - Score de regroupement (Distance maximale) : {score_max}")
        print(f"Cluster {i + 1} - Score de regroupement (Distance moyenne) : {score_avg}")
regroupement()

def separation():
    distance=[]
    dists = euclidean_distances(centroids)
    for i in range(-1,k-1):
        distance.append(dists[i][i+1])
    print("Entre les centres de clusters, la distance minimale = ",min(distance), "la distance maximale = ", max(distance), "moyenne = ", sum(distance)/len(distance))
separation()