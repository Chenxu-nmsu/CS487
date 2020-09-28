# import libraries
import myhierarchical_sklearn, mykmeans, myhierarchical_scipy, myDBSCAN
import argparse
from time import process_time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing

##################
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
###################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_approach', help='cluster_approach', choices=["kmeans", "hierarchicalSklearn", "hierarchicalScipy", 'DBSCAN','kmeans_elbow','DBSCAN_best_values'])
    parser.add_argument('data', help='Dataset', choices=['iris.txt', 'MNIST.csv'])

    # parameters in kmeans
    parser.add_argument('--n_clusters1', help='n_clusters in [kmeans]', type=int)
    parser.add_argument('--init', help='init in [kmeans]', type=str)
    parser.add_argument('--n_init', help='n_init in [kmeans]', type=int)
    parser.add_argument('--max_iter', help='max_iter in [kmeans]', type=int)

    # parameters in kmeans-elbow
    parser.add_argument('--K', help='K in [kmeans-elbow]', type=int)

    # parameters in hierarchicalSklearn
    parser.add_argument('--n_clusters2', help='n_clusters in [hierarchicalSklearn]', type=int)
    parser.add_argument('--affinity', help='affinity in [hierarchicalSklearn]', type=str)
    parser.add_argument('--linkage', help='linkage in [hierarchicalSklearn]', type=str)

    # parameters in hierarchicalScipy
    parser.add_argument('--method', help='method in [hierarchicalScipy]', type=str)
    parser.add_argument('--criterion', help='criterion in [hierarchicalScipy]', type=str)
    parser.add_argument('--k', help='k in [hierarchicalScipy]', type=int)

    # parameters in DBSCAN
    parser.add_argument('--eps', help='eps in [DBSCAN]', type=float)
    parser.add_argument('--min_samples', help='min_samples in [DBSCAN]', type=int)
    parser.add_argument('--metric', help='metric in [DBSCAN]', type=str)

    # parameters in DBSCAN - Best_values
    parser.add_argument('--K1', help='metric in [DBSCAN - Best_values]', type=int)

    args = parser.parse_args()
    args = vars(parser.parse_args())

    if args['data'] == "iris.txt":
        # import Iris dataset
        iris = pd.read_csv(args['data'], header=None)
        X = iris.iloc[:, 0:-1]
        y = iris.iloc[:, -1]

        # set labels for the y
        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(y)

    elif args['data'] == "MNIST.csv":
        mnist = pd.read_csv(args['data'])
        X = mnist.iloc[:, 0:-1]
        y = mnist.iloc[:, -1].to_numpy()

    else:
        print('Invalid Dataset Input')

    # record initial time
    start = process_time()

    # cluster approach selection
    if args['cluster_approach'] == "kmeans":
        km = mykmeans.km.fit(X)
        y_km = km.fit_predict(X)
        print(y_km)
        print('SSE = %.3f' % km.inertia_)

    elif args['cluster_approach'] == "kmeans_elbow":
        # ---> elbow approach
        from sklearn.cluster import KMeans
        distortions = []
        # Calculate distortions
        for i in range(1, args['K']):
            km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
            km.fit(X)
            distortions.append(km.inertia_)
        # Plot distortions for different K
        plt.figure(figsize=(4, 3))
        plt.plot(range(1, args['K']), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()
        plt.savefig('elbow.png', dpi=300)
        plt.show()

    elif args['cluster_approach'] == "hierarchicalSklearn":
        cluster = myhierarchical_sklearn.cluster.fit(X)
        cluster_labels = cluster.fit_predict(X)
        print(cluster_labels)

    elif args['cluster_approach'] == "hierarchicalScipy":
        row_cluster1 = myhierarchical_scipy.linkage(X, method = args['method'], metric = 'euclidean')
        clusters2 = myhierarchical_scipy.fcluster(row_cluster1, args['k'], criterion = args['criterion'])
        print(clusters2)

    elif args['cluster_approach'] == "DBSCAN":
        y_db = myDBSCAN.db.fit_predict(X)
        print(y_db)

    elif args['cluster_approach'] == "DBSCAN_best_values":
        db = myDBSCAN.Best_values(X, args['K1'])

    else:
        print('Invalid Cluster Approach')

    # calculate the time elapse
    elapse = process_time() - start

    print('\n#####  Result of {} cluster #####'.format(args['cluster_approach']))
    print('Running time: {1:.5f} s'.format(args['cluster_approach'], elapse))
    print('##############  End ###############\n')