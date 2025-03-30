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
seed_val = 2022
np.random.seed(seed_val)
tf.random.set_seed(seed_val)
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
# parameters for neural network
learning_rate = 1e-3
epochs = 200
batch_size = 32
num_classes = 2
# setup the tensorboard
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[1], "datasets")
print(cusum_result_folder)
current_file = file_path.stem
print(current_file)
logdir_base = Path("tensorboard_logs")
logdir = Path(logdir_base, current_file)
num_models = n_len * num_N


# %% main double for loop
num_loops = 0
for i in range(n_len):
    n = n_vec[i]
    print(n, i)
    for j in range(num_N):
        N = int(N_vec[j] / 2)

        # generate the dataset for alternative hypothesis
        np.random.seed(seed_val)  # use the current seed value
        tf.random.set_seed(seed_val)  # use the current seed value
        result = DataGenAlternative(
            N_sub=N,  # Use the calculated N_change_point
            B=B[i],
            mu_L=mu_L,
            n=n,
            ARcoef=rho,
            tau_bound=tau_bound,
            B_bound=B_bound,
            ar_model='Gaussian',
            sigma=1
        )
        data_alt = result["data"]
        tau_alt = result["tau_alt"]
        mu_R_alt = result["mu_R_alt"]
        # generate dataset for null hypothesis
        data_null = GenDataMean(N, n, cp=None, mu=(mu_L, mu_L), sigma=1)  # Use calculated N_no_change
        data_all = np.concatenate((data_alt, data_null), axis=0)
        y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
        tau_all = np.concatenate((tau_alt, np.repeat(0, N)), axis=0)
        mu_R_all = np.concatenate((mu_R_alt, np.repeat(mu_L, N)), axis=0)
        # generate the training dataset and test dataset
        x_train, y_train, tau_train, mu_R_train = shuffle(
            data_all, y_all, tau_all, mu_R_all, random_state=42
        )

        for num_layers in [1, 5, 10]:
            for k in range(3):
                m = m_mat[i, k]
                model_name = f"n{n}N{2 * N}m{m}l{num_layers}cpd"  # Include p in model_name
                print(model_name)
                model = general_simple_nn(n, num_layers, m=[m]*num_layers, num_classes=num_classes, model_name=model_name)
                model.summary()
                lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    learning_rate, decay_steps=10000, decay_rate=1, staircase=False
                )

                size_histories = {}
                epochdots = tfdocs.modeling.EpochDots()
                size_histories[model_name] = compile_and_fit(
                    model,
                    x_train,
                    y_train,
                    batch_size,
                    lr_schedule,
                    name=model_name,
                    log_dir=logdir,
                    max_epochs=epochs,
                    epochdots=epochdots
                )
                plotter = tfdocs.plots.HistoryPlotter(
                    metric='accuracy', smoothing_std=10
                )
                plt.figure(figsize=(10, 8))
                plotter.plot(size_histories)
                acc_name = model_name + f'+acc_func.png'  # Include p in acc_name
                acc_path = Path(logdir, model_name, acc_name)
                plt.savefig(acc_path)
                plt.clf()
                model_path = Path(logdir, model_name, 'model.keras')
                print(model_path)
                model.save(model_path)

end_time = time.time()
print(f"Total time taken: {end_time - begin_time} seconds")



