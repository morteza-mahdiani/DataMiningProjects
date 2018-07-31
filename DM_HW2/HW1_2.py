# show graphs with respect to Correlation
import math
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
import pandas as pd
from sklearn.utils import shuffle

# load data

training = pd.read_csv("train.csv", header= None, names=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
test = pd.read_csv("test.csv", header=None, names=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

# shuffling data

training = shuffle(training, random_state = 0)
test = shuffle(test, random_state = 0)

# plot

c1 = training['A']
c2 = training['B']
c3 = training['C']
c4 = training['D']
c5 = training['E']
c6 = training['F']
c7 = training['G']
tc7 = test['G']

# plt.plot(c7, c1, 'ro')
# plt.ylabel("target feature")
# plt.xlabel("firth column")
# plt.show()

# deal with missing values
# two strategies are explained in report but here we use mean

training['A'][training['A'] == 0] = None
training['A'].fillna(training['A'].mean(), inplace=True)

training['B'][training['B'] == 0] = None
training['B'].fillna(training['B'].mean(), inplace=True)

training['C'][training['C'] == 0] = None
training['C'].fillna(training['C'].mean(), inplace=True)

training['D'][training['D'] == 0] = None
training['D'].fillna(training['D'].mean(), inplace=True)

training['E'][training['E'] == 0] = None
training['E'].fillna(training['E'].mean(), inplace=True)

training['F'][training['F'] == 0] = None
training['F'].fillna(training['F'].mean(), inplace=True)

c7[c7 == 0] = None
c7.fillna(c7.mean(), inplace=True)

# training, validation and test
# slices

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
# model = linear_model.Lasso(alpha=0.1)
# model = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2,learning_rate=0.01, loss='ls')
model = linear_model.Ridge (alpha = .5)
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

# test LinearRegression

# model = linear_model.LinearRegression()
# test = test.drop('G', axis= 1)
# model.fit(training.drop('G', axis= 1), c7)
# print "LinearRegression predicted values for test dataSet:"
# print model.predict(test)

#test Lasso

# model = linear_model.Lasso(alpha=0.1)
# test = test.drop('G', axis= 1)
# model.fit(training.drop('G', axis= 1), c7)
# print "Lasso predicted values for test dataSet:"
# print model.predict(test)

#test Gradient Boosting method

# model = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2,learning_rate=0.01, loss='ls')
# test = test.drop('G', axis= 1)
# model.fit(training.drop('G', axis= 1), c7)
# print "GradientBoostingRegressor predicted values for test dataSet:"
# print model.predict(test)

#test Ridge

model = linear_model.Ridge (alpha = .5)
test = test.drop('G', axis= 1)
model.fit(training.drop('G', axis= 1), c7)
tst = model.predict(test)
# print "Ridge predicted values for test dataSet:"
# print tst

newTest = pd.read_csv("test.csv", header=None, names=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
newTest['G'] = tst

print newTest

newTest.to_csv("P1_submission.csv")