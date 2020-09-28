#########################################
########### Sample codes ################
----------------------------------------
########## iris.txt dataset ##########
### -> kmeans cluster
python main.py kmeans iris.txt --n_clusters1 3 --init random --n_init 200
python main.py kmeans iris.txt --n_clusters1 3 --init k-means++ --n_init 200
python main.py kmeans iris.txt --n_clusters1 3 --init k-means++ --n_init 500

### -> kmeans - elbow
python main.py kmeans_elbow iris.txt --K 10

### -> hierarchicalSklearn cluster
python main.py hierarchicalSklearn iris.txt --n_clusters2 3 --affinity euclidean --linkage complete
python main.py hierarchicalSklearn iris.txt --n_clusters2 3 --affinity manhattan --linkage complete
python main.py hierarchicalSklearn iris.txt --n_clusters2 3 --affinity euclidean --linkage single

### -> hierarchicalScipy cluster
python main.py hierarchicalScipy iris.txt --k 2 --method complete --criterion maxclust
python main.py hierarchicalScipy iris.txt --k 2 --method single --criterion maxclust
python main.py hierarchicalScipy iris.txt --k 2 --method single --criterion distance

### -> DBSCAN cluster
python main.py DBSCAN iris.txt --eps 0.2 --min_samples 5 --metric euclidean
python main.py DBSCAN iris.txt --eps 0.5 --min_samples 5 --metric euclidean
python main.py DBSCAN iris.txt --eps 0.2 --min_samples 10 --metric euclidean

### -> DBSCAN_best_values
python main.py DBSCAN_best_values iris.txt --K1 10

-------------------------------------
########## MNIST.csv dataset ##########
### -> kmeans cluster
python main.py kmeans MNIST.csv --n_clusters1 3 --init random --n_init 200
python main.py kmeans MNIST.csv --n_clusters1 3 --init k-means++ --n_init 200
python main.py kmeans MNIST.csv --n_clusters1 3 --init k-means++ --n_init 500

### -> kmeans - elbow
python main.py kmeans_elbow MNIST.csv --K 20

### -> hierarchicalSklearn cluster
python main.py hierarchicalSklearn MNIST.csv --n_clusters2 3 --affinity euclidean --linkage complete
python main.py hierarchicalSklearn MNIST.csv --n_clusters2 3 --affinity manhattan --linkage complete
python main.py hierarchicalSklearn MNIST.csv --n_clusters2 3 --affinity euclidean --linkage single

### -> hierarchicalScipy cluster
python main.py hierarchicalScipy MNIST.csv --k 2 --method complete --criterion maxclust
python main.py hierarchicalScipy MNIST.csv --k 2 --method single --criterion maxclust
python main.py hierarchicalScipy MNIST.csv --k 2 --method single --criterion distance

### -> DBSCAN cluster
python main.py DBSCAN MNIST.csv --eps 0.2 --min_samples 5 --metric euclidean
python main.py DBSCAN MNIST.csv --eps 0.5 --min_samples 5 --metric euclidean
python main.py DBSCAN MNIST.csv --eps 0.2 --min_samples 10 --metric euclidean

### -> DBSCAN_best_values
python main.py DBSCAN_best_values MNIST.csv --K1 20
