import time
begin_time = time.time()
import pathlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from keras import layers, losses, metrics, models
# %%
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import sys
import os

# Get the absolute path of the project root
script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Go up two levels

# Append the project root to sys.path
sys.path.append(project_root)

# Now, import the module
from autocpd.utils import *
from autocpd.neuralnetwork import *

seed = 2022  # Example seeds
np.random.seed(seed)
tf.random.set_seed(seed)
# All code below will be executed for each seed.

# %% parameter settings
n_vec = np.array([100], dtype=np.int32)  # the length of time series
n_len = len(n_vec)
epsilon = 0.05
Q = np.array(np.floor(np.log2(n_vec / 2)), dtype=np.int32) + 1
# the number of hidden nodes
m_mat = np.c_[3 * np.ones((n_len,), dtype=np.int32), 4 * Q, 2 * n_vec - 2]
N_vec = np.arange(100, 800, 100, dtype=np.int32)  # the sample size
num_N = len(N_vec)
B = np.sqrt(8 * np.log(n_vec / epsilon) / n_vec)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.5, 1.5])
rho = 0.0
sigma = 1
# parameters for neural network
learning_rate = 1e-3
epochs = 200
batch_size = 32
num_classes = 2
#  setup the tensorboard
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[2], "datasets", "CusumResult")
current_file = file_path.stem
print(current_file)
logdir_base = Path("tensorboard_logs", f"Trial_Seed")
logdir = Path(logdir_base, current_file + f"seed{seed}")
num_models = n_len * num_N
d = {
    'n': np.repeat(0, num_models),
    'N': np.repeat(0, num_models),
    'B': np.repeat(0.0, num_models),
    'Threshold': np.repeat(0.0, num_models)
}
Cusum_th = pd.DataFrame(data=d)

# %% main double for loop
num_loops = 0
for i in range(n_len):
    n = n_vec[i]
    print(n, i)
    for j in range(num_N):
        N = int(N_vec[j] / 2)
    
        #  generate the dataset for alternative hypothesis
        np.random.seed(seed)  # use the current seed value
        tf.random.set_seed(seed)  # use the current seed value
        result = DataGenAlternative(
            N_sub=N, # Use the calculated N_change_point
            B=B[i],
            mu_L=mu_L,
            n=n,
            ARcoef=rho,
            tau_bound=tau_bound,
            B_bound=B_bound,
            ar_model='Gaussian',
            sigma=sigma
        )
        data_alt = result["data"]
        tau_alt = result["tau_alt"]
        mu_R_alt = result["mu_R_alt"]
        #  generate dataset for null hypothesis
        data_null = GenDataMean(N, n, cp=None, mu=(mu_L, mu_L), sigma=1) # Use calculated N_no_change
        data_all = np.concatenate((data_alt, data_null), axis=0)
        y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
        tau_all = np.concatenate((tau_alt, np.repeat(0, N)), axis=0)
        mu_R_all = np.concatenate((mu_R_alt, np.repeat(mu_L, N)), axis=0)
        #  generate the training dataset and test dataset
        x_train, y_train, tau_train, mu_R_train = shuffle(
            data_all, y_all, tau_all, mu_R_all, random_state=42
        )
        # CUSUM, find the optimal threshold based on the training dataset

        # Compute the theoretical threshold under iid case as a start.
        threshold_star_theoretical = np.sqrt(2 * np.log(n / epsilon))
        y_cusum_train_max = np.apply_along_axis(MaxCUSUM, 1, x_train)
        threshold_candidate = threshold_star_theoretical * np.arange(
            0.1, 3, 0.05
        )
        mer = np.repeat(0.0, len(threshold_candidate))
        for ind, th in enumerate(threshold_candidate):
            y_pred_cusum_train = y_cusum_train_max > th
            conf_mat = tf.math.confusion_matrix(y_pred_cusum_train, y_train)
            mer[ind] = (conf_mat[0, 1] + conf_mat[1, 0]) / N/2 # MER calculated on total N

        threshold_opt = threshold_candidate[np.argmin(mer)]
        Cusum_th.at[num_loops, 'n'] = n
        Cusum_th.at[num_loops, 'N'] = 2 * N
        Cusum_th.at[num_loops, 'B'] = B[i]
        Cusum_th.at[num_loops, 'Threshold'] = threshold_opt
        num_loops += 1


# %%
# save the cusum threshold to folder datasets/CusumResult/
file_path = Path(__file__)
print(file_path)
print(file_path.parents[1])
pkl_path = Path(file_path.parents[1], "Thresholds", f"Thresholds_seed{seed}.pkl")
Cusum_th.to_pickle(pkl_path)
np.save(pkl_path, Cusum_th)
print(f"pickle path: {pkl_path}")
end_time = time.time()
print(f"Total time taken: {end_time - begin_time} seconds")
print(f"Cusum_th: {Cusum_th}")