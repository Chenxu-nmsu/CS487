from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')

def Best_values(X_test, K):
    # Small example of calculating the k−dists and plot them import numpy as np
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import numpy as np

    # Step 1: calculate the k distances
    # K = len(X_test) - 1
    # K = 10
    n = len(X_test)
    nn = NearestNeighbors(n_neighbors=(K + 1))
    nbrs = nn.fit(X_test)
    print(nbrs)
    distances, indices = nbrs.kneighbors(X_test)
    print(indices)
    print(distances)

    # Step 2: Sorted k−dist graph
    distanceK = np.empty([K, n])  # 1NN, 2NN, 3NN, ... , (K−1)th NN); each NN contains K distances for K points

    for i in range(K):  # for k−distances : sort them in descending order
        distance_Ki = distances[:, (i + 1)]
        distance_Ki.sort()
        distance_Ki = distance_Ki[:: -1]   # reverse distance_K1
        distanceK[i] = distance_Ki
    print(distanceK)
    # Step 3: plot the K distances to decide minPts and Epsilon

    plt.figure(figsize=(6, 4))
    for i in range(K):
        plt.plot(distanceK[i], label='K=%d' % (i + 1))
    plt.ylabel('distance')
    plt.xlabel('points')
    plt.legend(loc='best', ncol=3)
    plt.tight_layout()
    plt.savefig('Best_values.png', dpi=300)
    plt.show()
