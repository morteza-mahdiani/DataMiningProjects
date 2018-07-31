import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import sys
import time
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.cluster import normalized_mutual_info_score



plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


water_treatment = pd.read_csv("water-treatment.data.txt")
data_length = len(list(water_treatment))

headers = []
for i in range(data_length):
    headers.append(str(i))

water_treatment.columns = headers

water_treatment = pd.DataFrame(water_treatment.drop(water_treatment.columns[0], axis=1))

for header in list(water_treatment):
    for i in range(len(water_treatment[header])-1 , -1 , -1):
        if(water_treatment[header].iloc[i] == "?"):
            water_treatment = water_treatment.drop(water_treatment.index[[i]])

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    print (labels)
    return
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    # plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)



best_k  , best_score = 0  , 0
k = []
scores = []
for n_clusters in range( 2 , 20):
    kmean = KMeans(n_clusters=n_clusters).fit(water_treatment)
    labels = kmean.labels_.ravel()
    score =  metrics.silhouette_score(water_treatment, labels, metric='euclidean')
    k.append(n_clusters)
    scores.append(score)
    if(best_score < score):
        best_score = score
        best_k = n_clusters


# plt.figure(figsize=(12, 9))
# ax = plt.subplot(312)
# ax.stem(k,scores, '.')
# plt.show()
# print scores , k
print (best_score , best_k)
# print "###\n###\n###\n###\n###\n###\n###\n"

print ("#################")
# sys.exit()
htru_data = pd.read_csv("HTRU_2.csv")
data_length = len(list(htru_data))
headers = []

for i in range(data_length):
    headers.append(str(i))

htru_data.columns = headers
htru_data = pd.DataFrame(htru_data.drop(htru_data.columns[0], axis=1))

# agglomerativeClustering = AgglomerativeClustering(linkage='complete').fit(htru_data)
#
plot_clusters(htru_data, AgglomerativeClustering, (), {'linkage':'complete'})

normalized_mutual_info_score(labels)

plot_clusters(htru_data, AgglomerativeClustering, (), {'linkage':'ward'})




plt.show()
labels = kmean.labels_.ravel()
print (labels)