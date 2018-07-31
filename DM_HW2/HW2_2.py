# house pricing
import numpy as np
import math
from StdSuites.Type_Names_Suite import string
from sklearn import linear_model, ensemble
import pandas as pd
from sklearn.feature_selection.rfe import RFE
from sklearn.preprocessing.label import LabelEncoder
from sklearn.svm.classes import SVC
from sklearn.utils import shuffle
# load data
training = pd.read_csv("part2/train.csv" )
test = pd.read_csv("part2/test.csv" )
# shuffling data
training = shuffle(training, random_state = 0)
test = shuffle(test, random_state = 0)

# deal with missing values
test.fillna(0, inplace=True)
for clm in training.columns:
    if type(training[clm]) is (np.int64 or np.float):
        training[clm].fillna(training[clm].mean(), inplace=True)
    else:
        training[clm].fillna(training[clm].value_counts().idxmax(), inplace=True)
    if type(training[clm]) is string:
        training[clm].valuse = training[clm].astype(np.int64).values

# str to numeric
for clm in training.columns:
    if type(training[clm]) is not(np.int64 or np.float):
        training[clm] = LabelEncoder().fit_transform(training[clm])
for clm in test.columns:
    if type(test[clm]) is not(np.int64 or np.float):
        test[clm] = LabelEncoder().fit_transform(test[clm])

# training, validation and test
# slices
c7 = training["SalePrice"]
k = 10
slices = []
slicesOfc7 = []
for i in xrange(k):
    p = training[i::k]
    slices.append(p)
for i in xrange(k):
    p = c7[i::k]
    slicesOfc7.append(p)

# validation
# model = linear_model.LinearRegression()
# model = linear_model.Lasso(alpha= 0.0001)
# model = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2,learning_rate=0.01, loss='ls')
model = linear_model.Ridge (alpha = 0.5)
RMSE = 0
for i in range(10):
    x = [slices[(i+1) % 9], slices[(i+2) % 9], slices[(i+3) % 9], slices[(i+4) % 9], slices[(i+5) % 9], slices[(i+6) % 9], slices[(i+7) % 9], slices[(i+8) % 9]]
    x = pd.concat(x)
    y = [slicesOfc7[(i+1) % 9], slicesOfc7[(i+2) % 9], slicesOfc7[(i+3) % 9], slicesOfc7[(i+4) % 9], slicesOfc7[(i+5) % 9], slicesOfc7[(i+6) % 9], slicesOfc7[(i+7) % 9], slicesOfc7[(i+8) % 9]]
    y = pd.concat(y)
    model.fit(x, y)
    pr = model.predict(slices[i])
    RMSE += math.pow(((pr - slicesOfc7[i]) ** 2).sum() / len(slicesOfc7[i]), 0.5)
print "RMSE for validation is:"
print RMSE/k

# run SVM for extra features
svc = SVC(kernel="linear", C=1)
sel = RFE(estimator=svc, n_features_to_select=20, step=0.5, verbose=5)
sel.fit(training, c7)
training = training[:][sel.get_support(True)]
# after SVM
print "afteeeeeeeeeeeeeeeeeeeeeeer SVM:"
modelLR = linear_model.LinearRegression()
modelL = linear_model.Lasso(normalize=True)
modelGBR = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2,learning_rate=0.01, loss='ls')
modelR = linear_model.Ridge(alpha = .5)
RMSELR = 0
RMSEL = 0
RMSEGBR = 0
RMSER = 0
for i in range(10):
    x = [slices[(i+1) % 9], slices[(i+2) % 9], slices[(i+3) % 9], slices[(i+4) % 9], slices[(i+5) % 9], slices[(i+6) % 9], slices[(i+7) % 9], slices[(i+8) % 9]]
    x = pd.concat(x)
    y = [slicesOfc7[(i+1) % 9], slicesOfc7[(i+2) % 9], slicesOfc7[(i+3) % 9], slicesOfc7[(i+4) % 9], slicesOfc7[(i+5) % 9], slicesOfc7[(i+6) % 9], slicesOfc7[(i+7) % 9], slicesOfc7[(i+8) % 9]]
    y = pd.concat(y)
    modelLR.fit(x, y)
    modelL.fit(x, y)
    modelGBR.fit(x, y)
    modelR.fit(x, y)
    RMSELR += math.pow(((modelLR.predict(slices[i]) - slicesOfc7[i]) ** 2).sum() / len(slicesOfc7[i]), 0.5)
    RMSEL += math.pow(((modelL.predict(slices[i]) - slicesOfc7[i]) ** 2).sum() / len(slicesOfc7[i]), 0.5)
    RMSEGBR += math.pow(((modelGBR.predict(slices[i]) - slicesOfc7[i]) ** 2).sum() / len(slicesOfc7[i]), 0.5)
    RMSER += math.pow(((modelR.predict(slices[i]) - slicesOfc7[i]) ** 2).sum() / len(slicesOfc7[i]), 0.5)
print "RMSE for validation linear regression model is:"
print RMSELR/k
print "RMSE for validation lasso model is:"
print RMSEL/k
print "RMSE for validation gradient boosting regression is:"
print RMSEGBR/k
print "RMSE for validation ridge is:"
print RMSER/k

# fit test data and training data
for i in test.columns:
    if i not in training.columns:
        test = test.drop(i, axis= 1)

# test with Lasso model
model = linear_model.Ridge()
model.fit(training.drop("SalePrice", axis=1), c7)
tst = model.predict(test)
print "Ridge predicted values for test dataSet:"
print tst

# appropriate output
newTest = pd.DataFrame({
    "Id": pd.read_csv("part2/test.csv")["Id"],
    "SalePrice": tst
})
newTest.to_csv("P2_submission.csv", index=False)