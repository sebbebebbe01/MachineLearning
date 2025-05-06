import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
plt.rcParams.update({'font.size': 13})

def load_data():
    base_path = ""
    mat_file_path = base_path + "A2_data.mat"
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: File '{mat_file_path}' not found.")
        mat_data = None

    if mat_data is not None:
        # Access variables from the .mat file
        test_data = mat_data['test_data_01']
        test_labels = mat_data['test_labels_01']
        train_data = mat_data['train_data_01']
        train_labels = mat_data['train_labels_01']
        return [test_data, test_labels, train_data, train_labels]

### PCA
[test_data, test_labels, train_data, train_labels] = load_data()

## Normalise the data
X_train = train_data - np.mean(train_data,1).reshape(-1,1)
t_train = train_labels

n, N_train = X_train.shape # n = 28x28, originial dimensionality. N_train = number of data points
C = 1/N_train * X_train@X_train.T # Covariance matrix

start = time.time()

d = 2

(U, S, Vh) = np.linalg.svd(C) # Might as well use the covariance matrix instead of X_train since we only care about U
pc_vectors = U[:,:d]
X = pc_vectors.T @ X_train

end = time.time()
print(end-start)

plotting_data = pd.DataFrame({'pc1' : X[0,:], 'pc2' : X[1,:], 't' : t_train.reshape(1,-1)[0]})
palette = sns.color_palette("bright", 2)
sns.scatterplot(data=plotting_data, x='pc1', y='pc2', style='t', hue = 't', palette=palette)
plt.legend()
plt.title(r'PCA visualisation of dataset, $d=2$')
plt.show()