import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os

# Get the absolute path of the project root
script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
project_root = os.path.abspath(os.path.join(script_dir, "../.."))  # Go up two levels

# Append the project root to sys.path
sys.path.append(project_root)

# Now, import the module
from autocpd.utils import *

np.random.seed(2023)
tf.random.set_seed(2023)

# parameter settings
n_vec = np.array([100], dtype=np.int32)  # the length of time series
n_len = len(n_vec)
epsilon = 0.05
N_test = 100  # the sample size
N_vec = np.arange(100, 800, 100, dtype=np.int32)  # the sample size
B = np.sqrt(8 * np.log(n_vec / epsilon) / n_vec)
mu_L = 0
tau_bound = 2
num_repeat = 1
B_bound = np.array([0.25, 1.75])
rhos = 0
num_N = len(N_vec)

# setup the tensorboard and result folders
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[2], "datasets", "CNN")
current_file = file_path.stem
current_file = current_file.replace('Predict', 'Train')
logdir_base = Path("tensorboard_logs", f"CNN")
logdir = Path(logdir_base, current_file + f"seed{seed_val}")

# load the cusum thresholds
pkl_path = Path(cusum_result_folder, current_file + f'seed{seed_val}CNN.pkl')
Cusum_th = pd.read_pickle(pkl_path)
num_models = Cusum_th.shape[0]

# prediction
N = int(N_test * num_repeat / 2)
result_cusum = np.empty((num_models, num_repeat, 5))

for i in range(num_models):
    n = Cusum_th.at[i, 'n']
    N_train = Cusum_th.at[i, 'N']
    B_val = Cusum_th.at[i, 'B']
    threshold_opt = Cusum_th.at[i, 'Threshold']
    print("Model", i, "n:", n, "B:", B_val, "Threshold:", threshold_opt)

    # generate the dataset for alternative hypothesis
    np.random.seed(seed_val)  # numpy seed fixing
    tf.random.set_seed(seed_val)  # tensorflow seed fixing
    result = DataGenAlternative(
        N_sub=N,
        B=B_val,
        mu_L=mu_L,
        n=n,
        ARcoef=rhos,
        tau_bound=tau_bound,
        B_bound=B_bound,
        ar_model="AR0"
    )
    data_alt = result["data"]
    # generate dataset for null hypothesis
    data_null = GenDataMean(N, n, cp=None, mu=(mu_L, mu_L), sigma=1)
    data_all = np.concatenate((data_alt, data_null), axis=0)
    y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
    # generate the training dataset and test dataset
    data_all, y_all = shuffle(data_all, y_all, random_state=42)

    # CUSUM prediction
    y_cusum_test_max = np.apply_along_axis(MaxCUSUM, 1, data_all)
    y_pred_cusum_all = y_cusum_test_max > threshold_opt

    for j in range(num_repeat):
        # CUSUM
        ind = range(N_test * j, N_test * (j + 1))
        y_test = y_all[ind, 0]
        y_pred_cusum = y_pred_cusum_all[ind]
        confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_cusum)
        mer_cusum = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
        result_cusum[i, j, 0] = mer_cusum
        result_cusum[i, j, 1:] = np.reshape(confusion_mtx, (4,))

# save the cusum thresholds and prediction results
cusum_vec = np.mean(result_cusum, axis=1, keepdims=False)[:, 0]

plt.figure(figsize=(10, 8))
plt.plot(N_vec, cusum_vec, linewidth=4, marker='o', markersize=14)
plt.legend(['CUSUM'], fontsize=25)
plt.xlabel('N', fontsize=25)
plt.ylabel('MER Average', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('MER Average vs N (change in mean)', fontsize=25)

# Save figure
cm_name = current_file + f'MERAverage.png'
latex_figures_folder = Path(file_path.parents[2], "Figures")
mode = "CNN"
subfolder_name = mode
subfolder_path = latex_figures_folder / subfolder_name
subfolder_path.mkdir(parents=True, exist_ok=True)
figure_path = subfolder_path / cm_name
save_path = figure_path.parent / (figure_path.stem + f"seed{seed_val}{mode}.png")
plt.savefig(save_path, format='png')

print('CUSUM', cusum_vec)
# Save CUSUM results
path_cusum = Path(cusum_result_folder, current_file + f"_result_cusum_seed{seed_val}func{func}{mode}.npy")
np.save(path_cusum, result_cusum)
