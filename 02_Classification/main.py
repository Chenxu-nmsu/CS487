#!/usr/bin/env python
# coding: utf-8

import argparse
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from time import process_time
from sklearn.preprocessing import StandardScaler
from Classifiers import *
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # add arguments to input specific classifier and data to execute certain group of codes.
    # arguments of parameter in each classifier are also added.
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier', help='Classifier Name', choices = ['Perceptron', "svm_linear", "svm_nonlinear","tree_model", "knn"])
    parser.add_argument('data', help='Data No.', choices = ['digits', 'data2'])
    parser.add_argument('--max_iter', help = "max_iter [Perceptron]", type= int)
    parser.add_argument('--eta0', help='eta0 in [Perceptron]', type=float)
    parser.add_argument('--C1', help='C in [svm_linear]', type=int)
    parser.add_argument('--C2', help='C in [svm_nonlinear]',type = int)
    parser.add_argument('--gamma', help='gamma in [svm_nonlinear]', type=float)
    parser.add_argument('--max_depth', help='max_depth in [tree_model]', type=int)
    parser.add_argument('--n_neighbor', help='n_neighbor in [knn]',type=int)
    args = parser.parse_args()
    args = vars(parser.parse_args())
    
    # dataset selection
    if args['data'] == "digits":

        # (1) import dataset #1 (digits)
        digits = datasets.load_digits()

        # (2) selecting features
        X = digits.data
        y = digits.target
        
        # (3) splitting training and test datasets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
    
    elif args['data'] == "data2":
        
        import pandas as pd
        train_data = pd.read_csv('subject1_ideal.log', header=None, sep='\t')
        test_data = pd.read_csv('subject1_self.log', header=None, sep='\t')

        # train data
        X_train = train_data.iloc[range(0, len(train_data)-1, 50), 2:-1]     # column 2 to 119
        y_train = train_data.iloc[range(0, len(train_data)-1, 50), -1]       # last column (y)

        # test data
        X_test = test_data.iloc[range(0, len(test_data)-1, 20), 2:-1]        # column 2 to 119
        y_test = test_data.iloc[range(0, len(test_data)-1, 20), -1]          # last column (y)

    else:
        print('Error dataset input')

    # (4) feature scaling
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # record initial time
    start = process_time()

    # Classifier selection
    if args['classifier'] == "Perceptron":        
        ppn.max_iter = args['max_iter']
        ppn.eta0 = args['eta0']
        ppn.fit(X_train_std, y_train)
        y_pred_train = ppn.predict(X_train_std)
        y_pred_test = ppn.predict(X_test_std)

    elif args['classifier'] == "svm_linear":
        svm_linear.C = args['C1']
        svm_linear.fit(X_train_std, y_train)
        y_pred_train = svm_linear.predict(X_train_std)
        y_pred_test = svm_linear.predict(X_test_std)

    elif args['classifier'] == "svm_nonlinear":
        svm_nonlinear.C = args['C2']
        svm_nonlinear.gamma = args['gamma']
        svm_nonlinear.fit(X_train_std, y_train)
        y_pred_train = svm_nonlinear.predict(X_train_std)
        y_pred_test = svm_nonlinear.predict(X_test_std)

    elif args['classifier'] == "tree_model":        
        tree_model.max_depth = args['max_depth']
        tree_model.fit(X_train, y_train)

        # X_test not X_test_std
        y_pred_train = tree_model.predict(X_train)
        y_pred_test = tree_model.predict(X_test)

    elif args['classifier'] == "knn":
        knn.n_neighbor = args['n_neighbor']
        knn.fit(X_train_std, y_train)
        y_pred_train = knn.predict(X_train_std)
        y_pred_test = knn.predict(X_test_std)

    else:
        print('Error classifier')

    # calculate the time elapse for each classifier
    elapse = process_time() - start

    print('\n#############  Result of {} classifier ##############'.format(args['classifier']))
    print('The running time of {0} classifier is {1:.5f} s'.format(args['classifier'],elapse))
    print('\nTraining data: ')
    print('Misclassified samples: {}'.format((y_train != y_pred_train).sum()))
    print('Accuracy for training data: {0:.3f}'.format(accuracy_score(y_train,y_pred_train)))
    print('\nTesting data: ')
    print('Misclassified samples: {}'.format((y_test != y_pred_test).sum()))
    print('Accuracy for testing data: {0:.3f}'.format(accuracy_score(y_test,y_pred_test)))
    print('##########################  End #############################\n')