#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import seaborn as sns


# # 1 Perceptron binary classifier

# In[2]:


class perceptron(object):
    
    """Perceptron classifier.
    
    Parameters ------------ 
    eta : float
        Learning rate (between 0.0 and 1.0) 
    n_iter : int
        Passes over the training dataset. 
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes 
    ----------- 
    w_ : 1d-array
        Weights after fitting. 
    errors_ : list
        Number of misclassifications (updates) in each epoch. 
    """
    
    # Initialization
    def __init__(self, eta=0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    # Training
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update!= 0.0)
            self.errors_.append(errors)
        return self
    
    # Prediction
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)


# # 2 Adaline binary classifier

# In[3]:


class adaline(object):
    
    # Initialization
    def __init__(self, eta=0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
                             
    
    # Training
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []
        self.misclassify = []

        for _ in range(self.n_iter):
        
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) 
            self.w_[0] += self.eta * errors.sum()
            error = (errors**2).sum() / 2.0 
            self.cost_.append(error)
            
            # count the number of misclassify cases
            misclassify = 0
            
            for xi, target in zip(X,y):
                
                # compare prediction with origianl label each interation.
                update = self.eta*(target - self.predict(xi))
                misclassify += int(update !=0 )
                
            # sum the number of misclassify cases
            self.misclassify.append(misclassify)
            

        return self
    
    # Prediction
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    #
    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)


# # 3 Stochastic Gradient Descent (SGD) binary classifier

# In[4]:


class sgd(object):
    # Initialization
    def __init__(self, eta=0.01, n_iter = 50, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    # Training
    def fit(self, X, y):
        self._initialize_weights(X.shape[1]) 
        self.cost_ = []
        self.errors_=[]
        
        for i in range(self.n_iter):
            
            # count the number of misclassify cases
            errors = 0
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target)) 
                
                # compare the prediction with original lable each interation 
                update = self.eta*(target - self.predict(xi))
                errors += int(update!= 0.0)
                
            # sum the number of misclassify cases
            self.errors_.append(errors)
            avg_cost = sum(cost)/len(y) 
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights""" 
        if not self.w_initialized:
            self._initialize_weights(X.shape[1]) 
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y): 
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y)) 
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers""" 
        self.rgen = np.random.RandomState(self.random_state) 
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,size=1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights""" 
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>= 0.0, 1, -1)


# # 4. Dataset #1: Iris
# url: https://www.cs.nmsu.edu/~hcao/teaching/cs487519/data/iris.data

# In[5]:


# Read data (only two features for 2D plot)
url = "https://www.cs.nmsu.edu/~hcao/teaching/cs487519/data/iris.data"
df = pd.read_csv(url, header=None)
y = df.iloc[0:150, 4].values                # Select the last column with names(labels)
y = np.where(y == "Iris-setosa", 1, -1)     # Convert the class labels to two integer
X = df.iloc[0:150, [0,2]].values            # Extract sepal length and petal length


# In[6]:


# Plot data
plt.scatter(X[0:50, 0],X[0:50, 1], color = 'red', marker = 'o', label ='setosa')
plt.scatter(X[50:100, 0],X[50:100, 1], color = 'blue', marker = 'x', label ='versicolor')
plt.scatter(X[100:150, 0],X[100:150, 1], color = 'yellow', marker = '*', label ='virgincia')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


# In[7]:


# Read data (all features, 4 columns)
url = "https://www.cs.nmsu.edu/~hcao/teaching/cs487519/data/iris.data"
df = pd.read_csv(url, header=None)
y = df.iloc[0:, 4].values                # Select the last column with names(labels)
y = np.where(y == "Iris-setosa", 1, -1)     # Convert the class labels to two integer
X = df.iloc[0:, 0:3].values            # Extract sepal length and petal length


# ## 4.1 Dataset #1: perceptron classifier

# In[8]:


ppn = perceptron(eta=0.1, n_iter = 10)
ppn.fit(X,y)

print('Number of misclassifications in each iteration is {}'.format(ppn.errors_))
accuray = [(1-(error / len(y)))*100  for error in ppn.errors_]
print('\nAccuracy in each iteration is {} {}'.format(accuray,"%"))

plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()


# ## 4.2 Dataset #1: adaline classifier

# In[9]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

ada1 = adaline(n_iter=10, eta=0.0001).fit(X, y)

print('Number of misclassifications in each iteration is {}'.format(ada1.misclassify))
accuray = [(1-(x / len(y)))*100  for x in ada1.misclassify]
print('\nAccuracy in each iteration is {} {}'.format(accuray,"%"))

ax[0].plot(range(1, len(ada1.cost_) + 1),np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = adaline(n_iter=10, eta=0.01).fit(X, y)

ax[1].plot(range(1, len(ada2.cost_) + 1),np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()


# In[10]:


# feature scaling
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std() 
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada3 = adaline(n_iter=10, eta=0.0001).fit(X, y)
ada3.fit(X_std, y)

plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o') 
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('Adaline - Learning rate 0.0001')
plt.show()


# ## 4.3 Dataset #1: sgd classifier tested

# In[11]:


sgd1 = sgd(n_iter=10, eta=0.01, random_state=1) 
sgd1.fit(X_std, y)

print('Number of misclassifications in each iteration is {}'.format(sgd1.errors_))
accuray = [(1-(error / len(y)))*100  for error in sgd1.errors_]
print('\nAccuracy in each iteration is {} {}'.format(accuray,"%"))

plt.plot(range(1, len(sgd1.cost_) + 1), sgd1.cost_, marker='o') 
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()


# # 5. Dataset #2: Wine Data Set
# url: https://archive.ics.uci.edu/ml/datasets/wine

# In[12]:


# Read data
df2 = pd.read_csv('wine.data', header=None, sep=',')

y1 = df2.iloc[0:59, 0].values                # Select the last column with names(labels)
y2 = df2.iloc[130:178, 0].values             # Select the last column with names(labels)
y = np.concatenate([y1,y2],axis=0)
y = np.where(y == 1, 1, -1)                  # Convert the class labels to two integer

X1 = df2.iloc[0:59, [2,7]].values            # Extract Ash, Nonflavanoid phenols
X2 = df2.iloc[130:178, [2,7]].values         # Extract Ash, Nonflavanoid phenols
X = np.concatenate([X1,X2],axis=0)

print(df2.head(5))

# Plot data
plt.scatter(X[0:59, 0],X[0:59, 1], color = 'red', marker = 'o', label ='1')
plt.scatter(X[59:, 0],X[59:, 1], color = 'blue', marker = '*', label ='3')
plt.xlabel('Ash')
plt.ylabel('Nonflavanoid phenols')
plt.legend(loc='upper right')
plt.show()


# ## 5.1 Dataset #2: perceptron classifier

# In[13]:


ppn = perceptron(eta=0.1, n_iter = 20)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()


# ## 5.2 Dataset #2: adaline classifier

# In[14]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ada = adaline(n_iter=100, eta=0.0001).fit(X, y)
ax.plot(range(1, len(ada.cost_) + 1),np.log10(ada.cost_), marker='o')
ax.set_xlabel('Epochs')
ax.set_ylabel('log(Sum-squared-error)')
ax.set_title('Adaline - Learning rate 0.0001')
plt.show()


# ## 5.3 Dataset #2: sgd classifier tested

# In[15]:


X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std() 
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

sgd1 = sgd(n_iter=15, eta=0.01, random_state=1) 
sgd1.fit(X_std, y)

plt.plot(range(1, len(sgd1.cost_) + 1), sgd1.cost_, marker='o') 
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()


# # 7 One-vs-Rest strategy + sgd

# ## 7.1 Iris data set

# In[16]:


url = "https://www.cs.nmsu.edu/~hcao/teaching/cs487519/data/iris.data"
df = pd.read_csv(url, header=None)

X = df.iloc[0:150, 0:3].values                   # Take all features as the input
y = df.iloc[0:150, 4].values                     # Select the last column with names(labels)

# case I
y2 = np.where(y == "Iris-setosa", 1, -1)         # Convert the class labels to two
sgd2 = sgd(n_iter=100, eta=0.0001, random_state=1) 
sgd2.fit(X, y2)

plt.plot(range(1, len(sgd2.cost_) + 1), sgd2.cost_, marker='o') 
plt.title('Classifier for Iris-setos')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# case II
y3 = np.where(y == "Iris-versicolor", 1, -1)    
sgd3 = sgd(n_iter=100, eta=0.0001, random_state=1) 
sgd3.fit(X, y3)

plt.plot(range(1, len(sgd3.cost_) + 1), sgd3.cost_, marker='o') 
plt.title('Classifier for Iris-versicolor')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# case III
y4 = np.where(y == "Iris-virginica", 1, -1)     
sgd4 = sgd(n_iter=100, eta=0.0001, random_state=1) 
sgd4.fit(X, y4)

plt.plot(range(1, len(sgd4.cost_) + 1), sgd4.cost_, marker='o') 
plt.title('Classifier for Iris-virginica')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()


# In[17]:


# define the decision_function to differentiate these three types (For Iris data set).
def decision_function(X):
    
    # Case for Iris-setos
    if sgd2.predict(X) == 1 and sgd3.predict(X) != 1 and sgd4.predict(X) != 1:
        return "Iris-setosa"
    
    # Case for Iris-versicolor
    elif sgd2.predict(X) != 1 and sgd3.predict(X) == 1 and sgd4.predict(X) != 1:
        return "Iris-versicolor"
    
    # Case for Iris-virginica
    elif sgd2.predict(X) != 1 and sgd3.predict(X) != 1 and sgd4.predict(X) == 1:
        return "Iris-virginica"
    
    # error prediction
    else:
        return "Error"


# In[18]:


# Test for the trained classifier
df = pd.read_csv(url, header=None)
X = df.iloc[range(0,150), 0:3].values      
y = df.iloc[range(0,150), 4].values                     

# Print the fist row with predicted lable, original label and True or False
print('{0: <30} {1: <30} {2: <30}'.format('Predicted label', 'Original label', 'Prediction: True or False'))

# count the number of right predictions
right_pred = 0

for i,j in zip(X,y):
    predict_result = decision_function(i)
    
    # tell whether the prediction is right or not?
    if predict_result == j:
        right_pred += 1
    
    print('{0: <30} {1: <30} {2}'.format(predict_result, j, predict_result == j)) 


# In[23]:


# Show the accuracy result
print("Accuracy of this calssifier for Iris data set is  {}%".format(right_pred/len(y)*100))


# ## 7.2 Wine data set

# In[19]:


# Data selection
df2 = pd.read_csv('wine.data', header=None, sep=',')

X = df2.iloc[0:, 1:13].values 
y = df2.iloc[0:, 0].values

a = X[0:,0]
b = X[0:,6]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(a,b,c=y, marker="^")


# In[20]:


# Read data
df2 = pd.read_csv('wine.data', header=None, sep=',')

X = df2.iloc[0:, [0,6]].values 
y = df2.iloc[0:, 0].values

# case I
y2 = np.where(y == 1, 1, -1)         
sgd2 = sgd(n_iter=100, eta=0.0001, random_state=1) 
sgd2.fit(X, y2)

plt.plot(range(1, len(sgd2.cost_) + 1), sgd2.cost_, marker='o') 
plt.title('Classifier for Type-1')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# case II
y3 = np.where(y == 2, 1, -1)     # Convert the class labels to two
sgd3 = sgd(n_iter=100, eta=0.0001, random_state=1) 
sgd3.fit(X, y3)

plt.plot(range(1, len(sgd3.cost_) + 1), sgd3.cost_, marker='o') 
plt.title('Classifier for Type-2')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# case III
y4 = np.where(y == 3, 1, -1)      # Convert the class labels to two
sgd4 = sgd(n_iter=100, eta=0.0001, random_state=1) 
sgd4.fit(X, y4)

plt.plot(range(1, len(sgd4.cost_) + 1), sgd4.cost_, marker='o') 
plt.title('Classifier for Type-3')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()


# In[21]:


# define the decision_function to differentiate these three types (For wine data set).
def decision_function1(X):
    
    # Case for Type-1
    if sgd2.predict(X) == 1 and sgd3.predict(X) != 1 and sgd4.predict(X) != 1:
        return "1"
    
    # Case for Type-2
    elif sgd2.predict(X) != 1 and sgd3.predict(X) == 1 and sgd4.predict(X) != 1:
        return "2"
    
    # Case for Type-3
    elif sgd2.predict(X) != 1 and sgd3.predict(X) != 1 and sgd4.predict(X) == 1:
        return "3"
    
    # error prediction
    else:
        return "Error"


# In[22]:


# Test for the trained classifier
df2 = pd.read_csv('wine.data', header=None, sep=',')

X = df2.iloc[0:, [0,6]].values 
y = df2.iloc[0:, 0].values

# Print the fist row with predicted lable, original label and True or False
print('{0: <30} {1: <30} {2: <30}'.format('Predicted label', 'Original label', 'Prediction: True or False'))

# count the number of right predictions
right_pred = 0

for i,j in zip(X,y):
    predict_result = decision_function1(i)
    
    # tell whether the prediction is right or not?
    if str(j) == predict_result:
        right_pred += 1
    
    print('{0: <30} {1: <30} {2}'.format(predict_result, j, str(j) == predict_result))


# In[24]:


# Show the accuracy result
print("Accuracy of this calssifier for wine data set is  {}%".format(right_pred/len(y)*100))


# In[ ]:





# In[ ]:




