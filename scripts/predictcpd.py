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
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Go up one levels

# Append the project root to sys.path
sys.path.append(project_root)

# Now, import the module
from autocpd.utils import *

# Set random seeds for reproducibility
np.random.seed(2022)
tf.random.set_seed(2022)

# %% parameter settings
n_vec = np.array([100], dtype=np.int32)  # the length of time series
n_len = len(n_vec)
epsilon = 0.05
N_test = 500  # the sample size
N_vec = np.arange(100, 800, 100, dtype=np.int32)  # the sample size
B = np.sqrt(8 * np.log(n_vec / epsilon) / n_vec)

mu_L = 0
tau_bound = 2
num_repeat = 5
B_bound = np.array([0.25, 1.75])
rhos = 0
num_N = len(N_vec)
n = n_vec[0]
# Setup the tensorboard and result folders
file_path = Path(__file__)
current_file = file_path.stem.replace('predictcpd', 'traincpd')
logdir_base = Path("tensorboard_logs")
logdir = Path(logdir_base, current_file)
print(logdir)


# %% prediction

results_nn = {}  # keys will be the number of layers, values a dict of three NN results
for l in [1, 5, 10]:
    results_nn[l] = {
        0: np.empty((num_N, num_repeat, 5)),
        1: np.empty((num_N, num_repeat, 5)),
        2: np.empty((num_N, num_repeat, 5))
    }
N = int(N_test * num_repeat / 2)
print(B)
for i in range(num_N):
    # Generate the dataset for alternative hypothesis
    N_train = int(N_vec[i] / 2)
    result = DataGenAlternative(
        N_sub=N,
        B=B,
        mu_L=mu_L,
        n=n,
        ARcoef=rhos,
        tau_bound=tau_bound,
        B_bound=B_bound,
        ar_model="Gaussian",
        sigma=1
    )
    data_alt = result["data"]
    # Generate dataset for null hypothesis
    data_null = GenDataMean(N, n, cp=None, mu=(mu_L, mu_L), sigma=1)
    data_all = np.concatenate((data_alt, data_null), axis=0)
    y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
    # Shuffle the dataset
    data_all, y_all = shuffle(data_all, y_all, random_state=42)

    # Load NN Classifiers
    Q = int(np.floor(np.log2(n)))
    m_vec = np.array([3, 4 * Q, 2 * n - 2])
    for num_layers in [1, 5, 10]:
        model_path_list = []
        for k in range(3):
            m = m_vec[k]
            model_name = f"n{n}N{2*N_train}m{m}l{num_layers}cpd"
            model_path = Path(logdir, model_name, 'model.keras')
            model_path_list.append(model_path)

        model0 = tf.keras.models.load_model(model_path_list[0])
        model1 = tf.keras.models.load_model(model_path_list[1])
        model2 = tf.keras.models.load_model(model_path_list[2])

        y_pred_NN_all0 = np.argmax(model0.predict(data_all), axis=1)
        y_pred_NN_all1 = np.argmax(model1.predict(data_all), axis=1)
        y_pred_NN_all2 = np.argmax(model2.predict(data_all), axis=1)

        for j in range(num_repeat):
            ind = range(N_test * j, N_test * (j + 1))
            
            y_test = y_all[ind, 0]

            # NN0
            y_pred_nn0 = y_pred_NN_all0[ind]
            confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn0)
            mer_nn0 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
            results_nn[num_layers][0][i, j, 0] = mer_nn0
            results_nn[num_layers][0][i, j, 1:] = np.reshape(confusion_mtx, (4,))
            # NN1
            y_pred_nn1 = y_pred_NN_all1[ind]
            confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn1)
            mer_nn1 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
            results_nn[num_layers][1][i, j, 0] = mer_nn1
            results_nn[num_layers][1][i, j, 1:] = np.reshape(confusion_mtx, (4,))
            # NN2
            y_pred_nn2 = y_pred_NN_all2[ind]
            confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn2)
            mer_nn2 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
            results_nn[num_layers][2][i, j, 0] = mer_nn2
            results_nn[num_layers][2][i, j, 1:] = np.reshape(confusion_mtx, (4,))
 
nn1_vec = np.mean(results_nn[1][1], axis=1, keepdims=False)[:, 0]
nn2_vec = np.mean(results_nn[5][1], axis=1, keepdims=False)[:, 0]
nn3_vec = np.mean(results_nn[10][1], axis=1, keepdims=False)[:, 0]
mean_mer = np.array([nn1_vec, nn2_vec, nn3_vec])
# Save NN results for each layer configuration:
for layers_config, nn_results in results_nn.items():
    path_nn = Path(logdir_base, f"result_nn{layers_config}.npy")
    np.save(path_nn, {
        "NN0": nn_results[0],
        "NN1": nn_results[1],
        "NN2": nn_results[2]
    })
plt.figure(figsize=(10, 8))
markers = ['o', 'v', '*']
for i in range(len(mean_mer)):
    plt.plot(N_vec, mean_mer[i, :], linewidth=4, marker=markers[i], markersize=14)
    plt.legend(['m1l1', 'm5l1', 'm10l1'], fontsize=25)
plt.xlabel('N', fontsize=25)
plt.ylabel('MER Average', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('MER Average vs N (change in mean)', fontsize=25)
plt.show()