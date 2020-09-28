import numpy as np

dataset = np.array([
            [1, 1.0,  1, 0.072,  1],
            [2, 2.0,  1, 0.072,  1],
            [3, 3.0,  1, 0.072,  1],
            [4, 4.0, -1, 0.072, -1],
            [5, 5.0, -1, 0.072, -1],
            [6, 6.0, -1, 0.072, -1],
            [7, 7.0,  1, 0.167,  1],
            [8, 8.0,  1, 0.167, -1],
            [9, 9.0,  1, 0.167, -1],
            [10, 10, -1, 0.072, -1]])
print(dataset.shape)

x = dataset[:, 1]
y = dataset[:, 2]
w = dataset[:, 3]
y_pred = dataset[:, 4]

print('\n### Step 2(c): Compute weighted error rate ###')
Error_rate = w.T @ (y != y_pred)
print('Error_rate ={}'.format(Error_rate))

print('\n### Step 2(d): Compute coefficient ###')
alpha = 0.5*np.log((1-Error_rate)/Error_rate)
print('Coefficient ={}'.format(alpha))

print('\n### Step 2(e): Updated weights ###')
w_updated = w.T * np.exp(-alpha * y * y_pred)
print('Updated weights ={}'.format(w_updated))

print('\n### Step 2(f): Normalize weights to sum to 1 ###')
w_updated_norm = w_updated / np.sum(w_updated)
print('Normalized updated weights ={}'.format(w_updated_norm))
