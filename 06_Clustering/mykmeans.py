from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init='random', n_init =10, max_iter=300, tol=1e-04, random_state=0)