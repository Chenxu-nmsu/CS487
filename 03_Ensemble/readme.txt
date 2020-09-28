# Example:
################################################################
python main.py 'Classifier' 'data' --para1 2 --para2 2 ...
################################################################

===============================================================
### Codes for digits ###
==============================================================
python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 1 --bootstrap1 True

python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 5 --min_sample_leaf2 2

python main.py Adaboost digits --n_estimators3 1000 --learning_rate 0.8
===============================================================
### Codes for data2 ###
===============================================================
python main.py Bagging data2 --n_estimators1 10 --max_samples1 1 --max_features1 1 --bootstrap1 True

python main.py Random_forest data2 --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 5 --min_sample_leaf2 2

python main.py Adaboost data2 --n_estimators3 100 --learning_rate 0.5
===============================================================

===============================================================
### parameter tuning Bagging for digits dataset ###
==============================================================
python main.py Bagging digits --n_estimators1 10 --max_samples1 10 --max_features1 10 --bootstrap1 True
python main.py Bagging digits --n_estimators1 50 --max_samples1 10 --max_features1 10 --bootstrap1 True
python main.py Bagging digits --n_estimators1 100 --max_samples1 10 --max_features1 10 --bootstrap1 True

python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 10 --bootstrap1 True
python main.py Bagging digits --n_estimators1 10 --max_samples1 10 --max_features1 10 --bootstrap1 True
python main.py Bagging digits --n_estimators1 10 --max_samples1 100 --max_features1 10 --bootstrap1 True

python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 1 --bootstrap1 True
python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 5 --bootstrap1 True
python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 10 --bootstrap1 True

python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 1 --bootstrap1 True
python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 1 --bootstrap1 False
===============================================================
### parameter tuning Random_forest for digits dataset ###
==============================================================
python main.py Random_forest digits --n_estimators2 10 --criterion2 'gini' --max_depth2 2 --min_samples_split2 2 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 50 --criterion2 'gini' --max_depth2 2 --min_samples_split2 2 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 100 --criterion2 'gini' --max_depth2 2 --min_samples_split2 2 --min_sample_leaf2 2

python main.py Random_forest digits --n_estimators2 50 --criterion2 'gini' --max_depth2 2 --min_samples_split2 2 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 2 --min_samples_split2 2 --min_sample_leaf2 2

python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 2 --min_samples_split2 2 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 2 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 20 --min_samples_split2 2 --min_sample_leaf2 2

python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 2 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 5 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 10 --min_sample_leaf2 2

python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 5 --min_sample_leaf2 2
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 5 --min_sample_leaf2 5
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 5 --min_sample_leaf2 10
===============================================================
### parameter tuning Adaboost for digits dataset ###
==============================================================
python main.py Adaboost digits --n_estimators3 10 --learning_rate 0.1
python main.py Adaboost digits --n_estimators3 50 --learning_rate 0.1
python main.py Adaboost digits --n_estimators3 100 --learning_rate 0.1

python main.py Adaboost digits --n_estimators3 100 --learning_rate 0.001
python main.py Adaboost digits --n_estimators3 100 --learning_rate 0.01
python main.py Adaboost digits --n_estimators3 100 --learning_rate 0.1
===============================================================

===============================================================
### Codes for Task 4.1 (Comparison with baseline ) ###
==============================================================
# Baseline classifier
python main.py Perceptron digits --max_iter 50 --eta0 0.1

# ensemble methods classifiers
python main.py Bagging digits --n_estimators1 10 --max_samples1 1 --max_features1 1 --bootstrap1 True
python main.py Random_forest digits --n_estimators2 50 --criterion2 'entropy' --max_depth2 10 --min_samples_split2 5 --min_sample_leaf2 2
python main.py Adaboost digits --n_estimators3 1000 --learning_rate 0.8
===============================================================

Notes:
# classifier: ['Bagging', "Random_forest", "Adaboost"]
# data: ['digits', 'data2']

# data2 is the dataset of mammographic_masses.data. A preprocess of the data concerning missing and wrong values has been conducted
# in the main.py.
