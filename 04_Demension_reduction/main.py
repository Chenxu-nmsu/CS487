# import libraries
import myPCA, myLDA, kernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import argparse
from sklearn import datasets
from time import process_time
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

################################################
# #This part of code is used to preprocess the MNIST data set and save a .csv file with less number of instances
# #Pre-process MNIST dataset
# fetch dataset
# mnist = fetch_openml('mnist_784')
# X = mnist.data
# y = mnist.target
#
# # get the subset (X_train, y_train)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.98, random_state=1, stratify=y)
# data = np.hstack((X_train, np.atleast_2d(y_train).T))
#
# # save as a csv file
# pd.DataFrame(data).to_csv("MNIST.csv")
##################################################

if __name__ == '__main__':

    # add arguments to input specific classifier and data to execute certain group of codes.
    # arguments of parameter in each classifier are also added.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_reduction', help='dim_reduction', choices=['pca', "lda", "kpca"])
    parser.add_argument('data', help='Dataset', choices=['iris', 'MNIST'])

    # parameters in pca
    parser.add_argument('--n_components1', help="n_components in [pca]", type=int)

    # parameters in lda
    parser.add_argument('--n_components2', help='n_components in [lda]', type=int)

    # parameters in kpca
    parser.add_argument('--n_components3', help='n_components in [kpca]', type=int)
    parser.add_argument('--kernel', help='kernel in [kpca]', type=str)
    parser.add_argument('--gamma', help='gamma in [kpca]', type=int)

    args = parser.parse_args()
    args = vars(parser.parse_args())

    if args['data'] == "iris":
        # import Iris dataset
        iris = pd.read_csv(args['data'] + '.txt', header=None)
        X = iris.iloc[:, 0:-1]
        y = iris.iloc[:, -1]

        # set labels for the y
        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(y)

        # get train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    elif args['data'] == "MNIST":
        # Dataset 2: MNIST
        mnist = pd.read_csv(args['data'] + '.csv')
        X = mnist.iloc[:, 0:-1]
        y = mnist.iloc[:, -1].to_numpy()

        # get train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    else:
        print('Error dataset input')

    # standardized the data
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # initiate a tree_model
    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=1)

    # initiate model evaluation parameters
    accuracy_train = []
    accuracy_test = []
    precision = []
    recall = []
    F1 = []

    # record initial time
    start = process_time()

    # dim_reduction approach selection
    if args['dim_reduction'] == "pca":
        # pca approach
        myPCA.pca.n_components = args['n_components1']
        X_train = myPCA.pca.fit_transform(X_train_std)

    elif args['dim_reduction'] == "lda":
        # lda approach
        myLDA.lda.n_components = args['n_components2']
        X_train = myLDA.lda.fit_transform(X_train_std, y_train)

    elif args['dim_reduction'] == "kpca":
        # kpca approach
        kernelPCA.kpca.n_components = args['n_components3']
        kernelPCA.kpca.kernel = args['kernel']
        kernelPCA.kpca.gamma = args['gamma']
        X_train = kernelPCA.kpca.fit_transform(X_train_std)

    else:
        print('Error dim_reduction approach')


    ### cross validation
    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

    for k, (train, test) in enumerate(kfold):
        # model training
        tree_model.fit(X_train[train], y_train[train])

        # predict
        y_pred_train_train = tree_model.predict(X_train[train])
        y_pred_train_test = tree_model.predict(X_train[test])

        # model evaluation
        accuracy_train.append(accuracy_score(y_train[train], y_pred_train_train))
        accuracy_test.append(accuracy_score(y_train[test], y_pred_train_test))
        precision.append(precision_score(y_train[test], y_pred_train_test, average='weighted'))
        recall.append(recall_score(y_train[test], y_pred_train_test, average='weighted'))
        F1.append(f1_score(y_train[test], y_pred_train_test, average='weighted'))

    # calculate the time elapse for each classifier
    elapse = process_time() - start

    print('\n#############  Result of {} approach ######################'.format(args['dim_reduction']))
    print('The running time of {0} + DT is {1:.5f} s'.format(args['dim_reduction'], elapse))
    print('\n-->Training data: ')
    print('Training accuracy: {0:>5.3} +/- {1:<5.3}'.format(np.mean(accuracy_train), np.std(accuracy_train)))
    print('\n-->Testing data: ')
    print('Testing accuracy:  {0:>5.3} +/- {1:<5.3}'.format(np.mean(accuracy_test), np.std(accuracy_test)))
    print('Testing precision: {0:>5.3} +/- {1:<5.3}'.format(np.mean(precision), np.std(precision)))
    print('Testing recall:    {0:>5.3} +/- {1:<5.3}'.format(np.mean(recall), np.std(recall)))
    print('Testing F1:        {0:>5.3} +/- {1:<5.3}'.format(np.mean(F1), np.std(F1)))
    print('##########################  End #############################\n')