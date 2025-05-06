import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sounddevice as sd
plt.rcParams.update({'font.size': 12})


## Task 5
# lambda_list = [1.00000000e-03, 2.78255940e-03, 7.74263683e-03, 2.15443469e-02,5.99484250e-02, 1.66810054e-01, 4.64158883e-01, 1.29154967e+00,3.59381366e+00, 1.00000000e+01]
# RMSE_val = [4.19067745, 3.92945428, 3.31077553, 2.63199386, 1.96061969, 1.60931964,1.34487784, 1.21356279, 1.3654856,  2.23153486]
# RMSE_est = [1.34389818e-03, 3.36123728e-03, 8.50844515e-03, 2.27747527e-02, 5.60345673e-02, 1.42906876e-01 ,3.47815239e-01, 7.76118946e-01,1.25853381e+00 ,2.18769915e+00]

# lambda_hat = lambda_list[np.argmin(RMSE_val)]

# plt.semilogx(lambda_list, RMSE_val, marker = 'o', label = r'RMSE$_{val}$')
# plt.semilogx(lambda_list, RMSE_est, marker = 'x', label = r'RMSE$_{est}$')
# plt.axvline(x = lambda_hat, ls='dashed', color = 'k', label = r'$\hat\lambda=' + str(round(lambda_hat,2))+r'$')
# plt.legend(loc = 'center left')
# plt.xlabel(r'$\lambda$')
# plt.ylabel('RMSE')
# plt.grid()
# plt.show()

## Task 6
# lambda_list = np.logspace(np.log10(0.0001), np.log10(0.5), 10)
# RMSE_val = [0.1652726,  0.09067292, 0.05111327, 0.03945997, 0.03427335, 0.04063191,0.05836297, 0.08875403, 0.1328518,  0.17663214]
# RMSE_est = [0.01012326, 0.01049425, 0.01185605, 0.01582988, 0.02425271, 0.0362106,0.05556429, 0.08734352, 0.13237335, 0.17655174]
# lambda_hat = lambda_list[np.argmin(RMSE_val)]

# plt.semilogx(lambda_list, RMSE_val, marker = 'o', label = r'RMSE$_{val}$')
# plt.semilogx(lambda_list, RMSE_est, marker = 'x', label = r'RMSE$_{est}$')
# plt.axvline(x = lambda_hat, ls='dashed', color = 'k', label = r'$\hat\lambda=' + str(round(lambda_hat,4))+r'$')
# plt.legend(loc = 'lower right')
# plt.xlabel(r'$\lambda$')
# plt.ylabel('RMSE')
# plt.grid()
# plt.show()

## Task 7 
fs = 8820

sound_data0 = np.load('sound_vectors.npy')['noisy_test']
sound_data = np.load('denoised_audio.npy')
sound_data1 = np.load('experimental_audio1e-2.npy')
sound_data2 = np.load('experimental_audio2e-2.npy')
sound_data7 = np.load('experimental_audio7e-3.npy')


sd.play(sound_data0, fs, blocking=True)
sd.play(sound_data, fs, blocking=True)
sd.play(sound_data7, fs, blocking=True)
sd.play(sound_data1, fs, blocking=True)
sd.play(sound_data2, fs, blocking=True)
