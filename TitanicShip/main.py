# code created by morteza mahdiani
# method = logistic regression
import pandas as pd
import numpy as np
from matplotlib.mlab import PCA
from sklearn import tree, model_selection
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.linear_model.logistic import LogisticRegression
# read data
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier

training = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Cabin, Sex, Embarked, ... should not be NAN
training["Pclass"].fillna(training["Pclass"].mode(), inplace=True)
training["Sex"].fillna(training["Sex"].mode(), inplace=True)
training["Age"].fillna(training["Age"].median(), inplace=True)
training["SibSp"].fillna(training["SibSp"].mode(), inplace=True)
training["Parch"].fillna(test["Parch"].mode(), inplace=True)
training["Fare"].fillna(training["Fare"].median(), inplace=True)
training["Embarked"].fillna('S', inplace=True)


test["Pclass"].fillna(test["Pclass"].mode(), inplace=True)
test["Sex"].fillna(test["Sex"].mode(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)
test["SibSp"].fillna(training["SibSp"].mode(), inplace=True)
test["Parch"].fillna(test["Parch"].mode(), inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)
test["Embarked"].fillna('S', inplace=True)
# convert to numbers
training["Embarked"].loc[training["Embarked"] == "Q"] = 1
training["Embarked"].loc[training["Embarked"] == "S"] = 2
training["Embarked"].loc[training["Embarked"] == "C"] = 3

training["Sex"].loc[training["Sex"] == "male"] = 1
training["Sex"].loc[training["Sex"] == "female"] = 2


test["Embarked"].loc[test["Embarked"] == "Q"] = 1
test["Embarked"].loc[test["Embarked"] == "S"] = 2
test["Embarked"].loc[test["Embarked"] == "C"] = 3

test["Sex"].loc[test["Sex"] == "male"] = 1
test["Sex"].loc[test["Sex"] == "female"] = 2
# seprate tag and data for training
trainingTag = training['Survived']
testID = test["PassengerId"]
# removing some feature
training = training.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# pca = PCA(n_components=None)
# pca.fit(training)
# print(pca.explained_variance_ratio_)

# kFold
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# ensemble
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
# model2 = DecisionTreeClassifier()
# estimators.append(('cart', model2))
# model3 = SVC()
# estimators.append(('svm', model3))
# model4 = tree.DecisionTreeClassifier()
# estimators.append(('svm', model4))
# model5 = RandomForestClassifier(n_jobs=40)
# estimators.append(('svm', model5))
# model6 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1, 1), random_state=2)
# estimators.append(('svm', model6))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, training, trainingTag, cv=kfold)
print(results.mean())
print results


# #----- logistic regression  Model  ------
lg = LogisticRegression()
lg = lg.fit(training, trainingTag)
predictedValues0 = lg.predict(test)
#
# #----- DecisionTree  Model  ------
# DecisionTree = tree.DecisionTreeClassifier()
# DecisionTree.fit(training,trainingTag)
# predictedValues1 = DecisionTree.predict(test)
#
# #----- RandomForest Model ------
# clf = RandomForestClassifier(n_jobs=40)
# clf.fit(training,trainingTag.values.ravel())
# predictedValues2 = clf.predict(test)
#
# #----- NeuralNetwork Model ------
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1, 1), random_state=2)
# clf.fit(training,trainingTag.values.ravel())
# predictedValues3 = clf.predict(test)

# print predictedValues1
# shape results for output file
PassengerId = np.array(testID).astype(int)
solution = pd.DataFrame(predictedValues0, PassengerId, columns=["Survived"])
solution.to_csv("solution.csv", index_label=["PassengerId"])
