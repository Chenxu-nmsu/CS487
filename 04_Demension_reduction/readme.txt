#########################################
########### Notes #######################
# ->1.
# In the main.py, code from line 17 to 31 is the data preprocess for MNIST dataset.
# The purpose is to obtain a sub dataset from the original dataset.
# The total number of instances is 1400, accounting for 2% of the original dataset.


#########################################
########### Sample codes ################

########## Iris dataset ##########
### -> pca approach
python main.py pca iris --n_components1 1
python main.py pca iris --n_components1 2
python main.py pca iris --n_components1 3

### -> lda approach
python main.py lda iris --n_components2 1
python main.py lda iris --n_components2 2
python main.py lda iris --n_components2 3

### -> kpca approach
python main.py kpca iris --n_components3 1 --kernel 'rbf' --gamma 5
python main.py kpca iris --n_components3 2 --kernel 'rbf' --gamma 5
python main.py kpca iris --n_components3 3 --kernel 'rbf' --gamma 5
python main.py kpca iris --n_components3 3 --kernel 'rbf' --gamma 5
python main.py kpca iris --n_components3 3 --kernel 'sigmoid' --gamma 5
python main.py kpca iris --n_components3 3 --kernel 'cosine' --gamma 5
python main.py kpca iris --n_components3 2 --kernel 'rbf' --gamma 5
python main.py kpca iris --n_components3 2 --kernel 'rbf' --gamma 10
python main.py kpca iris --n_components3 2 --kernel 'rbf' --gamma 15

########## MNIST dataset ##########
### -> pca approach
python main.py pca MNIST --n_components1 10
python main.py pca MNIST --n_components1 30
python main.py pca MNIST --n_components1 50

### -> lda approach
python main.py lda MNIST --n_components2 10
python main.py lda MNIST --n_components2 30
python main.py lda MNIST --n_components2 50

### -> kpca approach
python main.py kpca MNIST --n_components3 10 --kernel 'cosine' --gamma 50
python main.py kpca MNIST --n_components3 30 --kernel 'cosine' --gamma 50
python main.py kpca MNIST --n_components3 50 --kernel 'cosine' --gamma 50

python main.py kpca MNIST --n_components3 50 --kernel 'sigmoid' --gamma 50
python main.py kpca MNIST --n_components3 50 --kernel 'cosine' --gamma 50

python main.py kpca MNIST --n_components3 50 --kernel 'cosine' --gamma 50
python main.py kpca MNIST --n_components3 50 --kernel 'cosine' --gamma 100
python main.py kpca MNIST --n_components3 50 --kernel 'cosine' --gamma 150