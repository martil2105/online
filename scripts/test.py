import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
from pathlib import Path

# Project path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from autocpd.utils import *
window_length = 100
stream_length = 400
num_repeats = 2
sigma = 1
seed = 2022
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
def li_cusum(stream, window_length, threshold):
    num_windows = len(stream) - window_length + 1
    detection_times = []
    max_cusum_scores = []
    for i in range(num_windows):
        window_data = stream[i:i+window_length]
        cusum_stats = ComputeCUSUM(window_data)
        max_stat = np.max(np.abs(cusum_stats))
        print(f"max_stat: {max_stat}")
        max_cusum_scores.append(max_stat)
        print(f"max_cusum_scores: {max_cusum_scores}")
        if max_stat > threshold:
            detection_times.append(i+window_length)
    return detection_times, np.max(max_cusum_scores)



data_null = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
data_alt = DataGenAlternative(num_repeats, B_val, mu_L, stream_length, ARcoef=rhos, tau_bound=tau_bound, B_bound=B_bound, sigma=sigma, ar_model="Gaussian")

y_null = data_null 
y_alt = data_alt['data']
true_tau = data_alt['tau_alt']

detection_times, max_cusum_scores = li_cusum(y_alt[0], window_length, 100)
print(f"detection_times: {detection_times}")
print(f"max_cusum_scores: {max_cusum_scores}")
print(f"true_tau: {true_tau[0]}")
#y_pred_cusum = y_cusum_test_max > threshold_opt
#print(f"y_pred_cusum: {y_pred_cusum}")
plt.plot(y_alt[0])
plt.show()



