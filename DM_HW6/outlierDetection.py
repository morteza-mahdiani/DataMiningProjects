import numpy as np
import pandas as pd
from scipy.io.matlab.mio import loadmat
from sklearn.covariance.outlier_detection import EllipticEnvelope
from sklearn.metrics.cluster.supervised import v_measure_score
# load data
data = loadmat("cardio.mat")
for i in data:
 if '__' not in i and 'readme' not in i:
   np.savetxt(("/Users/morteza/PycharmProjects/DM_HW6/"+i+".csv"),data[i],delimiter=',')
trainData=pd.read_csv("X.csv")
trainLabel=pd.read_csv("Y.csv")
# EllipticEnvelope
model = EllipticEnvelope(contamination=.1, assume_centered=True)
model.fit(trainData)
predicted = model.predict(trainData)
print("v-measure: " + str(v_measure_score(trainLabel, predicted)))