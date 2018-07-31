import pandas as pd

from sklearn.cluster import KMeans

from sklearn.metrics import v_measure_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

data = pd.read_csv('HTRU_2.csv', header=None)
data = data.sample(frac=0.8)
y = data[8]
data = data.drop([8], axis=1)
# print(data.shape)
# print(y.tail())

v = list()
silhouette = list()
last = 21
for i in range(2, last):
    km = KMeans(n_clusters=i).fit(data)
    predict = km.predict(data)
    print(predict)
    v.append(v_measure_score(y, predict))
    silhouette.append(silhouette_score(data, predict))

print(v)
print(silhouette)

plt.plot(range(2, last), v, 'r-o', label='V-Measure')
plt.plot(range(2, last), silhouette, 'b-o', label='Silhouette')
plt.legend()
plt.show()
