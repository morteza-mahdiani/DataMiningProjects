import pandas as pd

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import normalized_mutual_info_score

data = pd.read_csv('HTRU_2.csv', header=None)
y = data[8]
data = data.drop([8], axis=1)

linkage = ['complete', 'ward']
for l in linkage:
    ac = AgglomerativeClustering(linkage=l)
    predict = ac.fit_predict(data, y)
    print(l, normalized_mutual_info_score(y, predict))
