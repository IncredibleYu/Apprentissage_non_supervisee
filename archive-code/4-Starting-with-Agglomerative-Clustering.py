import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = '../artificial/'
name="xclara.arff"

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



### FIXER la distance
# 
tps1 = time.time()
seuil_dist=10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
k=4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")



#######################################################################

def Center():
    labels = model.labels_
    # Calculer les centres des clusters
    unique_labels = np.unique(labels)
    cluster_centers = []
    for label in unique_labels:
        cluster_points = datanp[labels == label]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)
    cluster_centers = np.array(cluster_centers)
    return cluster_centers

def regroupement(centroids, k):
    # Afficher les scores de regroupement pour chaque point de données
    for i in range(k):
        distances = []
        for n in datanp[labels== i] :
            distances.append(metrics.euclidean_distances([n, centroids[i]])[0][1])
        score_min = min(distances)  # Distance minimale dans le cluster
        score_max = max(distances)  # Distance maximale dans le cluster
        score_avg = sum(distances)/len(distances)  # Distance moyenne dans le cluster
        print(f"Cluster {i + 1} - Score de regroupement (Distance minimale) : {score_min}")
        print(f"Cluster {i + 1} - Score de regroupement (Distance maximale) : {score_max}")
        print(f"Cluster {i + 1} - Score de regroupement (Distance moyenne) : {score_avg}")


def separation(centroids, k):
    distance=[]
    dists = metrics.euclidean_distances(centroids)
    for i in range(-1,k-1):
        distance.append(dists[i][i+1])
    print("Entre les centres de clusters, la distance minimale = ",min(distance), "la distance maximale = ", max(distance), "moyenne = ", sum(distance)/len(distance))



def find_best_clustering(data, linkage_method, max_clusters=None, max_threshold=None):

    if max_clusters is not None and max_threshold is not None:
        raise ValueError("Please provide either max_clusters or max_threshold, not both.")

    best_score = float('-inf')
    best_k = None
    best_threshold = None
    sils = []

    if max_clusters is not None:
        # Iterating over different numbers of clusters
        for k in range(2, max_clusters + 1):
            model = cluster.AgglomerativeClustering(linkage=linkage_method, n_clusters=k)
            model.fit(data)
            silhouette_avg = metrics.silhouette_score(data, model.labels_)
            sils.append(silhouette_avg)
            if silhouette_avg > best_score:  # Silhouette score le plus élevé le mieux
                best_score = silhouette_avg
                best_k = k
        # Figure du coefficient silhouette
        plt.scatter(range(2, max_clusters + 1), sils, s=8)
        plt.plot(range(2, max_clusters + 1), sils)
        plt.xlabel("Nombre de clusters(k)")
        plt.ylabel("Coefficient de silhouette")
        plt.show()
        # Figure des clusters coloré avec meilleure k
        model = cluster.AgglomerativeClustering(linkage=linkage_method, n_clusters=best_k)
        model.fit(data)
        regroupement(Center(), best_k)
        separation(Center(), best_k)
        plt.scatter(data[:, 0], data[:, 1], c=model.labels_, s=8)
        plt.title(f"Clustering agglomeratif ({linkage_method}, n_clusters={best_k}), Silhouette Score: {best_score}")
        plt.show()
        print("n label = ", max(model.labels_))
        print("nb clusters =", best_k, "runtime =", round((tps2 - tps1) * 1000, 2), "ms, Silhouette Score =", best_score)

    elif max_threshold is not None:
        # Iterating over different distance thresholds
        thresholds = range(1, max_threshold + 1)
        for threshold in thresholds:
            model = cluster.AgglomerativeClustering(linkage=linkage_method, distance_threshold=threshold, n_clusters=None)
            model.fit(data)
            if len(set(model.labels_)) < 2:
                silhouette_avg = float('-inf')
            else:
                silhouette_avg = metrics.silhouette_score(data, model.labels_)
            sils.append(silhouette_avg)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_threshold = threshold
        # Figure du coefficient silhouette
        plt.scatter(thresholds, sils, s=8)
        plt.plot(thresholds, sils)
        plt.xlabel("thresholds")
        plt.ylabel("Coefficient de silhouette")
        plt.show()
        # Figure des clusters coloré avec meilleure threshold
        model = cluster.AgglomerativeClustering(linkage=linkage_method, distance_threshold=best_threshold, n_clusters=None)
        model.fit(data)
        best_k = max(model.labels_)+1
        regroupement(Center(), best_k)
        separation(Center(), best_k)
        plt.scatter(data[:, 0], data[:, 1], c=model.labels_, s=8)
        plt.title(f"Clustering agglomeratif ({linkage_method}, distance_threshold={best_threshold}), nb_clusters = {best_k}, Silhouette Score: {best_score}")
        plt.show()
        print("Threshold =", best_threshold, "runtime =", round((tps2 - tps1) * 1000, 2), "ms, Silhouette Score =", best_score)

# Exemple d'utilisation

find_best_clustering(datanp, linkage_method = 'ward' , max_clusters = 20)
#find_best_clustering(datanp, linkage_method='average', max_clusters=None, max_threshold = 20)
#find_best_clustering(datanp, linkage_method = 'ward', max_clusters=None, max_threshold = 40)