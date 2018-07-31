from collections import Counter
import pandas as pd
from sklearn import preprocessing, tree, model_selection
from sklearn.datasets.samples_generator import make_classification
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from validate_email import validate_email
from sklearn.metrics.classification import accuracy_score, f1_score
import numpy as np
# from imblearn.ensemble import BalanceCascade
# ----------load data


test = pd.read_csv('X_test.csv')
yTrain = pd.read_csv('Y_train.csv')
xTrain = pd.read_csv('X_train.csv')
# ----------pre processing of datda

# finding correlated data
# corr_matrix =  xTrain.corr(method = 'pearson', min_periods=1)
# print corr_matrix[corr_matrix>.9]

# deleting correlated columns
del xTrain['hour_a']
del xTrain['amount']
del test['hour_a']
del test['amount']

# check validation of emails
# for e in xTrain['customerAttr_b']:
#     if validate_email(e):
#         xTrain['valid_mail'] = 1
#     else:
#         xTrain['valid_mail'] = 0
#
# for e in test['customerAttr_b']:
#     if validate_email(e):
#         test['valid_mail'] = 1
#     else:
#         test['valid_mail'] = 0
del xTrain['customerAttr_b']
del test['customerAttr_b']

# change strings to number
# label_encoder = preprocessing.LabelEncoder()
# xTrain['state'] = label_encoder.fit_transform(xTrain['state'].values)
# test['state'] = label_encoder.fit_transform(test['state'].values)

del xTrain['state']
del test['state']

# ----------oversampling and undersampling

# bc = BalanceCascade(random_state=42)
# xTrain, yTrain = bc.fit_sample(xTrain, yTrain)
# X_res, y_res = bc.fit_sample(xTrain, yTrain)
# print('Resampled dataset shape {}'.format(Counter(y_res[0])))


# -----------models
# kFold
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

# ensemble
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4 = tree.DecisionTreeClassifier()
estimators.append(('svm', model4))
model5 = RandomForestClassifier(n_jobs=40)
estimators.append(('svm', model5))
model6 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1, 1), random_state=2)
estimators.append(('svm', model6))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, xTrain, yTrain, cv=kfold)
print("Ensemble Result: " + str(results.mean()))
# logistic regression
LogisticRegression = LogisticRegression()
LogisticRegression.fit(xTrain,yTrain)
predictedValues0 = LogisticRegression.predict(test)

# DecisionTree  Model
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree.fit(xTrain,yTrain)
predictedValues1 = DecisionTree.predict(test)

# RandomForest Model
clf = RandomForestClassifier(n_jobs=40)
clf.fit(xTrain,yTrain.values.ravel())
predictedValues2 = clf.predict(test)

# NeuralNetwork Model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1, 1), random_state=2)
clf.fit(xTrain,yTrain.values.ravel())
predictedValues3 = clf.predict(test)

# ------------compute f1-score and accuracy
ac = accuracy_score(yTrain, predictedValues0)
fs = f1_score(yTrain, predictedValues0, average='macro')
print("Accuracy: "+str(ac))
print("F1-SCORE: "+str(fs))
# print predictedValues1