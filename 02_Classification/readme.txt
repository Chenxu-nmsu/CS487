# Example:
################################################################
python main.py 'Classifier' 'data' --para1 2 --para2 2 ...
################################################################

===============================================================
### Codes for digits
==============================================================
python main.py Perceptron digits --max_iter 50 --eta0 0.1

python main.py svm_linear digits --C1 1

python main.py svm_nonlinear digits --C2 10 --gamma 0.1

python main.py tree_model digits --max_depth 50

python main.py knn digits --n_neighbor 2
===============================================================
### Codes for data2
===============================================================
python main.py Perceptron data2 --max_iter 50 --eta0 0.1

python main.py svm_linear data2 --C1 1

python main.py svm_nonlinear data2 --C2 10 --gamma 0.1

python main.py tree_model data2 --max_depth 10

python main.py knn data2 --n_neighbor 2
===============================================================

==============================================================
## Codes for parameter analysis for digits data
==============================================================
python main.py Perceptron digits --max_iter 50 --eta0 0.1
python main.py Perceptron digits --max_iter 50 --eta0 0.001
python main.py Perceptron digits --max_iter 50 --eta0 0.00001
python main.py Perceptron digits --max_iter 10 --eta0 0.1
python main.py Perceptron digits --max_iter 50 --eta0 0.1
python main.py Perceptron digits --max_iter 100 --eta0 0.1

python main.py svm_linear digits --C1 1
python main.py svm_linear digits --C1 5
python main.py svm_linear digits --C1 10

python main.py svm_nonlinear digits --C2 1 --gamma 0.1
python main.py svm_nonlinear digits --C2 10 --gamma 0.1
python main.py svm_nonlinear digits --C2 20 --gamma 0.1
python main.py svm_nonlinear digits --C2 10 --gamma 0.1
python main.py svm_nonlinear digits --C2 10 --gamma 0.2
python main.py svm_nonlinear digits --C2 10 --gamma 0.5

python main.py tree_model digits --max_depth 10
python main.py tree_model digits --max_depth 50
python main.py tree_model digits --max_depth 100

python main.py knn digits --n_neighbor 2
python main.py knn digits --n_neighbor 5
python main.py knn digits --n_neighbor 10
==============================================================

Notes:
# classifier: ['Perceptron', "svm_linear", "svm_nonlinear","tree_model", "knn"]
# data: ['digits', 'data2']

# data2 is the dataset of subject1.
### You much put the following two log files in the same folder with the .py file ###
    1. train_data: subject1_ideal.log
    2. test_data: subject1_self.log
    All features collected by sensors [2:120] in these two datasets are used in training and testing.
    Since the original dataset is too large. The instances are picked in ranges of 50 and 20 in train_data and testing_data.
    Thus, all classifiers can be trained and tested during a reasonable period of time.

# if you want to input another dataset, you should change the 'data2' name in main.py file.
    The last column will be automatically # setted as the dataset label.