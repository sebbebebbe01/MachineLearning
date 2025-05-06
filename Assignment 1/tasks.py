"""
Module to run all tasks regarding first assignment.

How you manage this module is not as important. You can choose whether you do as is written in the
skeleton or if you prefer to have input-values from the command line. However, keep the names of
all methods in the lasso-module identical as we can then correct the assignment easier.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import sounddevice as sd
from scipy.signal.windows import hann
plt.rcParams.update({'font.size': 13})

def read_data(data_to_read):
    """
    Returns the specified data entries in the list of strings data_to_read
    Parameters:
    - data_to_read: A list containing any of the following strings:
        ['t', 'x', 'n', 'x_interp', 'n_interp', 't_test', 't_train', 'x_audio', 'fs']
    Returns:
    - data_dictionary: A dictionary with keys specified by data_to_read
    """
    with open("A1_data.npy", "rb") as f:
        t_test = np.load(f)
        t_train = np.load(f)
        x = np.load(f)
        x_audio = np.load(f)
        x_interp = np.load(f)
        fs = np.load(f)
        n = np.load(f)
        n_interp = np.load(f)
        t = np.load(f)

    all_data_dict = {'t_test' : t_test,
                       't_train' : t_train,
                       'x' : x,
                       'x_audio' : x_audio,
                       'x_interp' : x_interp,
                       'fs' : fs,
                       'n' : n,
                       'n_interp' : n_interp,
                       't' : t}
    
    data_dictionary = {k: all_data_dict[k] for k in data_to_read}

    return data_dictionary


def task4(lamb = 1.87, data = None, w_old = None):
    """
    Runs code for task 4
    """

    # print('λ = '+ str(lamb))



    ## If data not given as a param, load the necessary variables from the data file. This also allows plotting
    if data is None:
        plot = True
        data = read_data(['t', 'x', 'n', 'x_interp', 'n_interp'])
        t, X, n, X_interp, n_interp = data['t'], data['x'], data['n'], data['x_interp'], data['n_interp']
    else: # For tasks 5-7
        plot = False
        t, X = data['t'], data['x']

    [N,M] = X.shape # N = Number of data points, M = number of weights

    
    ## Constructing the weights via coordinate descent
    n_iter = 50 # The number of iterations to update the weights according to equation 3 in the manual
    update_cycle = 10 # When to update the zero-terms

    ## If no w_old is given as param, initialise as zero
    if w_old is None:
        w_old = np.zeros((M,1)) # Define an array to store the weights during updates
    
    ## Define preliminary residual
    r_i = t - X@w_old
    for j in range(n_iter):
        ## Update the zero-weights only every update_cycle nr of iterations.
        ## Randomise the order of weight updates.
        if j % update_cycle and j>1:
            ind_nonzero = np.nonzero(w_old>lamb)[0]
            randind = np.random.permutation(len(ind_nonzero))
            indvec_random = ind_nonzero[randind]
        else:
            indvec_random = np.random.permutation(M)
        
        for idx in indvec_random: # Loop over all weights
            # Define our x_i and r_i for this iteration
            x_i = X[:,idx:idx+1]
            w_i_old = w_old[idx]

            ## Inefficient method of calculating r
            # r_i = t - X[:,indvec_random[:i]] @ w_new[indvec_random[:i]] - X[:,indvec_random[i+1:]] @ w_old[indvec_random[i+1:]] # This can be made more efficient!
            ## More efficient method, add and remove individual terms.
            r_i += x_i * w_i_old
            
            # Equation 3 in the manual
            if np.abs(x_i.T @ r_i) <= lamb:
                w_i_new = 0
            else:
                xTr = x_i.T@r_i
                xTx = np.sum(x_i**2)
                w_i_new = (xTr - np.sign(xTr) * lamb) / xTx
            
            ## Remove x_i multiplied by the UPDATED weight
            r_i -= x_i*w_i_new
            ## Add to new weight vector
            w_old[idx] = w_i_new
        

    # ### Count the number of non-zero weights
    # nr_of_non_zero = np.count_nonzero(w_new)
    # print('Number of non-zero weights:')
    # print(nr_of_non_zero)

    ## Plotting
    if plot:
        y = X@w_old # The reconstructed data points

        plt.scatter(n,y, label='Reconstructed')
        plt.plot(n_interp, X_interp@w_old, label='Interpolated')
        plt.scatter(n,t, label='Target')
        plt.legend()
        plt.xlabel('Time')
        plt.title(r'$\lambda='+str(round(lamb,2))+r'$')
        plt.show()
    return w_old

def task5(data = None, w_old = None, K = 5, lambda_list = None):
    """
    Runs code for task 5
    """

    ## Load the necessary variables from the data file if no data-param is included, this allows plotting
    if data is None:
        plot = True
        data = read_data(['t', 'x'])
    else:
        plot = False
    t, X = data['t'], data['x']
    [N,M] = X.shape # N = Number of data points (50), M = number of weights (1000)

    ## Definitions and initialisations
    if lambda_list is None:
        min_lambda = 0.001
        max_lambda = 10
        N_lambda = 100
        lambda_list = np.logspace(np.log10(min_lambda), np.log10(max_lambda), N_lambda)
    else:
        N_lambda = len(lambda_list)

    SE_val = np.zeros((K,N_lambda))
    SE_est = np.zeros((K,N_lambda))


    ## Cross-validation indexing
    random_ind = np.random.permutation(N) # Select random indices for validation and estimation
    N_val = N//K # How many samples per fold

    for k_fold in range(K):

        print('k-fold iteration number: ' + str(k_fold+1) + '/' + str(K))
    
        val_ind = random_ind[k_fold*N_val : (k_fold+1)*N_val] # Select validation indices
        est_ind = np.setdiff1d(random_ind, val_ind) # Select estimation indices

        if w_old is None:
            w_old = np.zeros((M,1)) # Initialize estimate for warm-starting.

        for j, lambda_j in enumerate(lambda_list):
            # Split the data into training and validation
            print('Hyperparameter nr: ' + str(j))
            data_est = {'x' : X[est_ind], 't' : t[est_ind]}
            data_val = {'x' : X[val_ind], 't' : t[val_ind]}

            # Perform coordinate descent (pass it on to the previous task)
            w_hat = task4(lambda_j, data = data_est, w_old = w_old)

            SE_val[k_fold, j] = 1/N_val * np.linalg.norm(data_val['t'] - data_val['x']@w_hat)**2
            SE_est[k_fold, j] = 1/(N-N_val) * np.linalg.norm(data_est['t'] - data_est['x']@w_hat)**2

            w_old = w_hat


    ## Calculate the RMSE
    RMSE_val = np.sqrt(1/K * np.sum(SE_val, 0))
    RMSE_est = np.sqrt(1/K * np.sum(SE_est, 0))
    
    lambda_hat = lambda_list[np.argmin(RMSE_val)]

    if plot:
        print('Choice of lambda for lowest validation RMSE:')
        print('λ = ' + str(lambda_hat))

        plt.semilogx(lambda_list, RMSE_val, marker = 'o', label = r'RMSE$_{val}$')
        plt.semilogx(lambda_list, RMSE_est, marker = 'x', label = r'RMSE$_{est}$')
        plt.axvline(x = lambda_hat, ls='dashed', color = 'k', label = r'$\hat\lambda=' + str(round(lambda_hat,2))+r'$')
        plt.legend(loc = 'center left')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('RMSE')
        plt.grid()
        plt.show()

    return [w_old, SE_val, SE_est]



def task6():
    """
    Runs code for task 6
    """
    data = read_data(['t_train', 'x_audio'])
    t_train, X = data['t_train'], data['x_audio']
    [N,M] = X.shape # N = Number of data points (352), M = number of weights (2000)

    ## Definitions and initialisation
    N_frames = len(t_train) // N
    K = 5
    min_lambda = 0.0001
    max_lambda = 0.5
    N_lambda = 10
    lambda_list = np.logspace(np.log10(min_lambda), np.log10(max_lambda), N_lambda)

    w_opt = np.zeros((M, N_frames))
    SE_val = np.zeros((K, N_lambda))
    SE_est = np.zeros((K, N_lambda))

    w_old = np.reshape(w_opt[:,0], (M,1))
    ## Start iterating over each frame
    for frame in range(N_frames):
        print('Progress: ' + str(100*frame/N_frames) + ' %')

        t = t_train[frame*N : (frame+1)*N] # The targets for the current frame
        frame_data = {'x' : X, 't' : t} # Create dictionary to pass onto the previous task
   
        [w_new, SE_val_frame, SE_est_frame] = task5(frame_data, w_old, K, lambda_list) # k-fold cross-validation

        ## Add the squared error to the cumulative sum of errors (matrices)
        SE_val += SE_val_frame
        SE_est += SE_est_frame

        w_old = w_new
        w_opt[:,frame] = w_new.ravel()
    
    RMSE_val = np.sqrt(np.mean(SE_val, 0))
    RMSE_est = np.sqrt(np.mean(SE_est, 0))

    lambda_hat = lambda_list[np.argmin(RMSE_val)]
    print('Choice of lambda for lowest validation RMSE:')
    print('λ = ' + str(lambda_hat))

    ## Plotting
    plt.semilogx(lambda_list, RMSE_val, marker = 'o', label = r'RMSE$_{val}$')
    plt.semilogx(lambda_list, RMSE_est, marker = 'x', label = r'RMSE$_{est}$')
    plt.axvline(x = lambda_hat, ls='dashed', color = 'k', label = r'$\hat\lambda=' + str(round(lambda_hat,4))+r'$')
    plt.legend(loc = 'lower right')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('RMSE')
    plt.grid()
    plt.show()


def task7():
    """
    Runs code for task 7
    """
    lambda_hat = 0.0044 # As determined from the previous task
    lambda_exp = 0.01 # Results in less static noise, but also lower audio quality

    data = read_data(['t_test', 'x_audio', 'fs'])
    t_test, X, fs = data['t_test'], data['x_audio'], data['fs'][0][0]

    clean_test = lasso_denoise(t_test, X, lambda_hat) # The lambda resulting in lowest RMSE_val

    # np.save('denoised_audio.npy', clean_test)

    sd.play(clean_test, fs, blocking = True)



def main():
    """
    Runs a specified task given input from the user
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--task",
        choices=["4", "5", "6", "7"],
        help="Runs code for selected task.",
    )
    args = parser.parse_args()
    try:
        if args.task is None:
            task = 0
        else:
            task = int(args.task)
    except ValueError:
        print("Select a valid task number")
        return
    start = time.time()

    if task == 4:
        task4()
    elif task == 5:
        task5()
    elif task == 6:
        task6()
    elif task == 7:
        task7()
    else:
        raise ValueError("Select a valid task number")
    
    end = time.time()
    print('Elapsed time: ' + str(end-start) + ' s')


def lasso_denoise(Tnoisy, X, lambda_val):
    """
    Denoises the data in Tnoisy using LASSO estimates for hyperparameter lambdaopt.
    Cycles through the frames in Tnoisy, calculates the LASSO estimate, selecting
    the non-zero components and reconstructing the data using these components only,
    using a WOLS estimate, weighted by the Hanning window.

    Parameters:
    - Tnoisy (numpy.ndarray): NNx1 noisy data vector
    - X (numpy.ndarray): NxM regression matrix
    - lambda_val (float): Hyperparameter value (selected from cross-validation)

    Returns:
    - Yclean (numpy.ndarray): NNx1 denoised data vector
    """

    # Sizes
    NN = len(Tnoisy)
    N, M = X.shape

    # Frame indices parameters
    loc = 0
    hop = N // 2
    idx = np.arange(N)

    Z = np.diag(hann(N))  # Weight matrix
    Yclean = np.zeros_like(Tnoisy)  # Clean data preallocation

    while loc + N <= NN:
        t = Tnoisy[loc + idx]  # Pick out data in the current frame
        data = {'t' : t, 'x' : X}
        wlasso = task4(lambda_val, data)  # Calculate LASSO estimate
        nzidx = np.abs(wlasso.reshape(-1)) > 0  # Find nonzero indices

        # Calculate weighted OLS estimate for nonzero indices
        wols = np.linalg.lstsq(Z @ X[:, nzidx], Z @ t, rcond=None)[0]
        # Reconstruct denoised signal
        Yclean[loc + idx] += Z @ X[:, nzidx] @ wols

        loc += hop  # Move indices for the next frame
        print(f"{int(loc / NN * 100)} %")  # Show progress

    print("100 %")
    return Yclean


if __name__ == "__main__":
    main()