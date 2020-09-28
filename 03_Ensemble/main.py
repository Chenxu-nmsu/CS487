
import argparse
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from time import process_time
from sklearn.preprocessing import StandardScaler
import myBagging, myRandom_forest, myAdaboost

import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # add arguments to input specific classifier and data to execute certain group of codes.
    # arguments of parameter in each classifier are also added.
    parser = argparse.ArgumentParser()
    parser.add_argument('ensemble_approach', help='Ensemble Approach',
                        choices=['Bagging', "Random_forest", "Adaboost"])
    parser.add_argument('data', help='Data No.', choices=['digits', 'data2'])

    # parameters in BaggingClassifier
    parser.add_argument('--n_estimators1', help="n_estimators in [BaggingClassifier]", type=int)
    parser.add_argument('--max_samples1', help='max_sample in [RandomForestClassifier]', type=int)
    parser.add_argument('--max_features1', help='max_feature in [RandomForestClassifier]', type=int)
    parser.add_argument('--bootstrap1', help='bootstrap in [RandomForestClassifier]', type=bool)

    # parameters in RandomForestClassifier
    parser.add_argument('--n_estimators2', help='n_estimators in [RandomForestClassifier]', type=int)
    parser.add_argument('--criterion2', help='criterion in [RandomForestClassifier]', type=str)
    parser.add_argument('--max_depth2', help='max_depth in [RandomForestClassifier]', type=int)
    parser.add_argument('--min_samples_split2', help='min_samples in [RandomForestClassifier]', type=int)
    parser.add_argument('--min_sample_leaf2', help='min_sample_leaf in [RandomForestClassifier]', type=int)

    # parameters in AdaBoostClassifier
    parser.add_argument('--n_estimators3', help='n_estimators in [AdaBoostClassifier]', type=int)
    parser.add_argument('--learning_rate3', help='learning_rate in [AdaBoostClassifier]', type=float)

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    elif args['data'] == "data2":
        import pandas as pd

        # read data2
        df = pd.read_csv('mammographic_masses.data', header=None)
        df.columns = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
        '''
        Missing Attribute Values: 
        - BI-RADS assessment: 2 
        - Age: 5 
        - Shape: 31 
        - Margin: 48 
        - Density: 76 
        - Severity: 0
        '''
        # preprocess dataset with missing values and wrong values
        df['BI-RADS'].replace(['?', '55', '6', '0'], ['2', '2', '2', '2'], inplace=True)
        df['Age'].replace('?', '5', inplace=True)
        df['Shape'].replace('?', '31', inplace=True)
        df['Margin'].replace('?', '48', inplace=True)
        df['Density'].replace('?', '76', inplace=True)
        df['Severity'].replace('?', '0', inplace=True)

        X = df.iloc[:, 0:5]
        y = df.iloc[:, -1]

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

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
    if args['ensemble_approach'] == "Bagging":
        myBagging.bag.n_estimators = args['n_estimators1']
        myRandom_forest.forest.max_samples = args['max_samples1']
        myRandom_forest.forest.max_features = args['max_features1']
        myRandom_forest.forest.bootstrap = args['bootstrap1']

        myBagging.bag.fit(X_train_std, y_train)

        y_pred_train = myBagging.bag.predict(X_train_std)
        y_pred_test = myBagging.bag.predict(X_test_std)

    elif args['ensemble_approach'] == "Random_forest":
        myRandom_forest.forest.n_estimators = args['n_estimators2']
        myRandom_forest.forest.criterion = args['criterion2']
        myRandom_forest.forest.max_depth = args['max_depth2']
        myRandom_forest.forest.min_samples_split = args['min_samples_split2']
        myRandom_forest.forest.min_samples_leaf = args['min_sample_leaf2']


        myRandom_forest.forest.fit(X_train_std, y_train)

        y_pred_train = myRandom_forest.forest.predict(X_train_std)
        y_pred_test = myRandom_forest.forest.predict(X_test_std)

    elif args['ensemble_approach'] == "Adaboost":
        myAdaboost.ada.n_estimators = args['n_estimators3']
        myAdaboost.ada.learning_rate = args['learning_rate3']

        myAdaboost.ada.fit(X_train_std, y_train)
        y_pred_train = myAdaboost.ada.predict(X_train_std)
        y_pred_test = myAdaboost.ada.predict(X_test_std)

    else:
        print('Error classifier')

    # calculate the time elapse for each classifier
    elapse = process_time() - start

    print('\n#############  Result of {} classifier ##############'.format(args['ensemble_approach']))
    print('The running time of {0} classifier is {1:.5f} s'.format(args['ensemble_approach'], elapse))
    print('\nTraining data: ')
    print('Misclassified samples: {}'.format((y_train != y_pred_train).sum()))
    print('Accuracy for training data: {0:.3f}'.format(accuracy_score(y_train, y_pred_train)))
    print('\nTesting data: ')
    print('Misclassified samples: {}'.format((y_test != y_pred_test).sum()))
    print('Accuracy for testing data: {0:.3f}'.format(accuracy_score(y_test, y_pred_test)))
    print('##########################  End #############################\n')