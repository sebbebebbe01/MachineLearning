import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.svm as svm
plt.rcParams.update({'font.size': 13})

def K_means_clustering(X, K):
    """
    Perform K-means clustering on input data.

    Parameters:
    - X: numpy.ndarray
        DxN matrix of input data.
    - K: int
        Number of clusters.

    Returns:
    - y: numpy.ndarray
        Nx1 vector of cluster assignments.
    - C: numpy.ndarray
        DxK matrix of cluster centroids.
    """

    D, N = X.shape

    intermax = 50
    conv_tol = 1e-6

    # Initialize
    C = np.mean(X, axis=1).reshape(D, 1) + np.std(X, axis=1).reshape(D, 1) * np.random.randn(D, K)
    y = np.zeros(N)
    Cold = C.copy()

    for j in range(intermax):
        # Step 1: Assign to clusters
        y = []
        for i in range(N):
            x_i = X[:,i:i+1]
            d_i = fxdist(x_i, C)
            y.append(np.argmin(d_i))

        # Step 2: Assign new clusters
        nr_of_elements_in_clusters = np.zeros(K)
        accum_pos_in_cluster = np.zeros((D,K))
        for i, cluster in enumerate(y):
            nr_of_elements_in_clusters[cluster] += 1
            accum_pos_in_cluster[:,cluster] += X[:,i]

        C = accum_pos_in_cluster / nr_of_elements_in_clusters

        if fcdist(C, Cold) < conv_tol:
            return y, C

        Cold = C.copy()

    return y, C

def fxdist(x,C):
    diff = x - C
    d = np.linalg.norm(diff, ord=2, axis=0)
    # DO NOT CHANGE
    return d

def fcdist(C1,C2):
    """
    Computes a pairwise distance between the vectors in C1 and C2 using euclidian norm

    Parameters:
    - C1 & C2: numpy.ndarray
        DxK matrices of K centroid coordinates in D-space

    Returns:
    - d: float
        Distance between the two compared matrices
    """
    assert C1.shape == C2.shape, "The two centroid matrices need to have the same shape."
    
    diff = C1 - C2
    d = np.linalg.norm(diff)
    # DO NOT CHANGE
    return d


def load_data():
    # Replace '/path/to/file/' with the path to your .mat file
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

def PCA_visualisation(train_data, labels, centroids):
    """
    Uses PCA to reduce the dimensionality of the data to two dimensions to plot it in a scatter plot.
    The corresponding 2-dimensional location of the centroids is also shown.

    Parameters:
    - train_data: numpy.ndarray
        DxN matrix. Each column is an image.
    - labels: list
        Length N. The corresponding group of each image.
    - centroids: numpy.ndarray
        DxK matrix containing the location of each centroid.
    """
    ## Normalise the data
    X_train = train_data - np.mean(train_data,1).reshape(-1,1)


    n, N_train = X_train.shape # n = 28x28, originial dimensionality. N_train = number of data points
    C = 1/N_train * X_train@X_train.T # Covariance matrix
    K = centroids.shape[1]

    d = 2

    (U, S, Vh) = np.linalg.svd(C) # Might as well use the covariance matrix instead of X_train since we only care about U
    pc_vectors = U[:,:d]
    X = pc_vectors.T @ X_train
    projected_centroids = pc_vectors.T @ centroids

    plt.figure()

    plotting_data = pd.DataFrame({'pc1' : X[0,:], 'pc2' : X[1,:], 't' : labels})

    palette = sns.color_palette("bright", K)
    sns.scatterplot(data=plotting_data, x='pc1', y='pc2', style='t', hue = 't', palette=palette)
    for i in range(K):  # for each centroid
        plt.scatter(projected_centroids[0,i], projected_centroids[1,i],
                    color=palette[i], s=200, marker='X', edgecolor='k', linewidth=1.5)
    plt.legend()
    plt.title(r'PCA visualisation of K-means clustering, $K=' + str(K) + r'$')
    plt.show()

def centroid_visualisation(C):
    """
    Visualises the centroids C

    Parameters:
    - C: numpy.ndarray
        DxK matrix. Each column is a centroid.
    """
    
    D, K = C.shape

    fig, axs = plt.subplots(1, K)

    for k in range(K):
        c_k = C[:,k]
        M = c_k.reshape((int(D**0.5),int(D**0.5)))

        axs[k].imshow(M.T) # Transpose to rotate the images such that e.g. ones are vertical

    plt.show()

def K_means_classifier(data, labels, C):

    D, N = data.shape
    K = C.shape[1]
    labels = labels.ravel()

    cluster_labels = np.zeros((2,K)) # 2 for the two classes zero and one.

    for i in range(N):
        x_i = data[:,i:i+1]
        d_i = fxdist(x_i, C)
        y_i = np.argmin(d_i) # The asigned cluster based on the proximity d_i
        cluster_labels[labels[i], y_i] += 1

    cluster_acc = np.zeros(K)
    for k in range(K):
        cluster = cluster_labels[:,k]
        cluster_acc[k] = max(cluster) / sum(cluster)

    return cluster_acc

def support_vector_machine(train_data, train_labels, test_data, kernel, C, gamma = 0.014):
    """
    Runs sklearns linear support vector machine

    Parameters:
    - data: numpy.ndarray
        DxN matrix containing N data points of D dimensions.
    - labels: numpy.ndarray
        len = N. Binary labels for supervised learning
    - kernel: string
        e.g. 'linear' or 'rbf' 
    - C: float
    - gamma: float
        Scaling parameter for a Gaussian kernel
    """
    if kernel == 'rbf':
        kernelsvm = svm.SVC(kernel = kernel, gamma=gamma, C = C)
    elif kernel == 'linear':
        kernelsvm = svm.SVC(kernel = kernel, C = C)

    X_train = train_data.T
    X_test = test_data.T
    t_train = train_labels.ravel()

    kernelsvm.fit(X_train, t_train)

    y_train = kernelsvm.predict(X_train)
    y_test = kernelsvm.predict(X_test)

    return y_train, y_test

def present_SVM_results(y, labels):
    """
    Prints the results from the predicted labels y and the true labels in labels

    Parameters:
    - y: numpy.ndarray
    - labels: numpy.ndarray
    """
    
    labels = labels.ravel()

    misclass = 0
    results_matrix = np.zeros((2,2))
    for pred_class in range(2):
        indices = np.where(y==pred_class)
        true_labels = labels[indices]
        correct_class = sum(true_labels == pred_class)
        incorrect_class = len(indices[0]) - correct_class
        misclass += incorrect_class

        results_matrix[pred_class, pred_class] = correct_class
        results_matrix[pred_class, 1-pred_class] = incorrect_class

    print('Row 0: predicted class zero')
    print('Row 1: predicted class 1')
    print(results_matrix)
    print()
    print('Misclassification rate (%):')
    print(100*misclass/len(labels))
    

if __name__ == "__main__":
    data = load_data()
    [test_data, test_labels, train_data, train_labels] = data
    nbr_clusters = 2 # Replace with you chosen int

    ## Exercise 8
    y, C = K_means_clustering(train_data, nbr_clusters)
    PCA_visualisation(train_data, y, C)

    ## Exercise 9
    centroid_visualisation(C)

    ## Exercise 10/11
    train_accuracy = K_means_classifier(train_data, train_labels, C)
    test_accuracy = K_means_classifier(test_data, test_labels, C)
    print('Training data:')
    for k in range(nbr_clusters):
        print('Cluster ' + str(k) + ':')
        print('Accuracy - ' + str(train_accuracy[k]))
        print()

    print('Test data:')
    for k in range(nbr_clusters):
        print('Cluster ' + str(k) + ':')
        print('Accuracy - ' + str(test_accuracy[k]))
        print()

    ## Exercise 12/13
    kernel = 'rbf'
    gamma = 0.014 # Default: 1/(train_data.var()*D) = 0.014067
    C = 1.0
    y_train, y_test = support_vector_machine(train_data, train_labels, test_data, kernel, C, gamma)

    print('Training data')
    present_SVM_results(y_train, train_labels)

    print('#######')
    print('Test data')
    present_SVM_results(y_test, test_labels)
