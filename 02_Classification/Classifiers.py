#!/usr/bin/env python
# coding: utf-8

## All classifiers

# （1）Perceptron
from sklearn.linear_model import Perceptron
# def perceptron_func(max_iter, eta0):
ppn = Perceptron(max_iter=40, eta0=0.1,random_state=1)

# (2) SVM linear + nonlinear(RBF)
from sklearn.svm import SVC
svm_linear = SVC(kernel = 'linear', C=1.0, random_state=1)
svm_nonlinear = SVC(kernel = 'rbf', C=10.0, random_state=1, gamma=0.10)

# (3) decision tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=1)

# (4) K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, p=2, metric = 'minkowski')