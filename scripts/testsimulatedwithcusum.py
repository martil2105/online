import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from time import time
begin_time = time()

# Project path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from autocpd.utils import *
from sklearn.utils import shuffle

# Parameters
window_length = 100
num_repeats = 1000
stream_length = 2000
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
thresholds = [99999]

# Load model
current_file = "train"
model_name = "n100N600m24l5"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)

np.random.seed(seed)
tf.random.set_seed(seed)
#get cusum threshold
Cusum_th = np.load(Path(project_root, "Thresholds", f"Thresholds_seed2022.npy"))
thresholds_opt = Cusum_th[:,3]
threshold_opt = thresholds_opt[3]

# Batched version of detection function
def detect_change_in_stream_loc_batched(stream, model, window_length, threshold):
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1) 
    logits = model.predict(windows, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
    detection_idx = np.argmax(probs > threshold)
    if probs[detection_idx] > threshold:
        detection_time = detection_idx + window_length
    else:
        detection_time = 0
    return detection_time, np.max(probs)



# Generate null data and compute threshold percentile
data_null = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
num_streams = data_null.shape[0]
max_probabilities = []
max_probabilities_cusum = []
max_probabilities_logit_cusum = []
dt, max_prob = detect_change_in_stream_loc_batched(data_null[0], model, window_length, thresholds[0])
print(f"dt: {dt}, max_prob: {max_prob}")
for i in range(num_streams):
    dt, max_prob = detect_change_in_stream_loc_batched(data_null[i], model, window_length, thresholds[0])
    max_probabilities.append(max_prob)
    dt_cusum, max_prob_cusum = li_cusum(data_null[i], window_length, thresholds[0])
    max_probabilities_cusum.append(max_prob_cusum)
    dt_logit_cusum, max_prob_logit_cusum = detect_change_in_stream_batched_cusum(data_null[i], model, window_length, thresholds[0])
    max_probabilities_logit_cusum.append(max_prob_logit_cusum)
    print(f"i: {i}")
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99]
#regular
percentile_80 = np.percentile(max_probabilities, 80)
percentile_85 = np.percentile(max_probabilities, 85)
percentile_90 = np.percentile(max_probabilities, 90)
percentile_95 = np.percentile(max_probabilities, 95)
percentile_99 = np.percentile(max_probabilities, 99)
#logit cusum
percentile_80_logit_cusum = np.percentile(max_probabilities_logit_cusum, 80)
percentile_85_logit_cusum = np.percentile(max_probabilities_logit_cusum, 85)
percentile_90_logit_cusum = np.percentile(max_probabilities_logit_cusum, 90)
percentile_95_logit_cusum = np.percentile(max_probabilities_logit_cusum, 95)
percentile_99_logit_cusum = np.percentile(max_probabilities_logit_cusum, 99)
#cusum
percentile_80_cusum = np.percentile(max_probabilities_cusum, 80)
percentile_85_cusum = np.percentile(max_probabilities_cusum, 85)
percentile_90_cusum = np.percentile(max_probabilities_cusum, 90)
percentile_95_cusum = np.percentile(max_probabilities_cusum, 95)
percentile_99_cusum = np.percentile(max_probabilities_cusum, 99)
print(f"Repeats: {num_streams}, 95th percentile: {percentile_95}")
percentiles = [percentile_80,percentile_85,percentile_90, percentile_95, percentile_99]
percentiles_cusum = [percentile_80_cusum,percentile_85_cusum,percentile_90_cusum, percentile_95_cusum, percentile_99_cusum]
percentiles_logit_cusum = [percentile_80_logit_cusum,percentile_85_logit_cusum,percentile_90_logit_cusum, percentile_95_logit_cusum, percentile_99_logit_cusum]
# Estimate ARL (Average Run Length)
arl = np.zeros((len(percentiles),num_repeats))
arl_cusum = np.zeros((len(percentiles),num_repeats))
arl_logit_cusum = np.zeros((len(percentiles_logit_cusum),num_repeats))
data = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
num_streams = data.shape[0]
for idx, percentile in enumerate(percentiles):
    for i in range(num_streams):
        dt, _ = detect_change_in_stream_loc_batched(data[i], model, window_length, percentile)
        if dt > 0:
            arl[idx,i] = dt
        else:
            arl[idx,i] = stream_length
for idx, percentile in enumerate(percentiles_cusum):
    for i in range(num_streams):
        dt_cusum, _ = li_cusum(data[i], window_length, percentile)
        if dt_cusum > 0:
            arl_cusum[idx,i] = dt_cusum
        else:
            arl_cusum[idx,i] = stream_length
for idx, percentile in enumerate(percentiles_logit_cusum):
    for i in range(num_streams):
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data[i], model, window_length, percentile)
        if dt_logit_cusum > 0:
            arl_logit_cusum[idx,i] = dt_logit_cusum
        else:
            arl_logit_cusum[idx,i] = stream_length

print(f"arl_cusum: {arl_cusum}")
arl_cusum = np.mean(arl_cusum,axis=1)
arl = np.mean(arl,axis=1)
#arl_logit_cusum = np.mean(arl_logit_cusum,axis=1)
print(f"arl_logit_cusum: {arl_logit_cusum}")
print(f"arl_cusum: {arl_cusum}")

result_alt =  DataGenAlternative(
    
                N_sub=num_repeats,
                B=B_val,
                mu_L=mu_L,
                n=stream_length,
                ARcoef=rhos,
                tau_bound=tau_bound,
                B_bound=B_bound,
                ar_model="Gaussian",
                sigma=sigma
            )
data_alt = result_alt["data"]     

true_tau_alt = result_alt["tau_alt"] 
detection_delay = np.zeros((len(percentiles),num_repeats))
detection_delay_cusum = np.zeros((len(percentiles),num_repeats))
detection_delay_logit_cusum = np.zeros((len(percentiles_logit_cusum),num_repeats))
fp_cusum = np.zeros(len(percentiles))
fn_cusum = np.zeros(len(percentiles))
fp = np.zeros(len(percentiles))
fn = np.zeros(len(percentiles))
fp_logit_cusum = np.zeros(len(percentiles_logit_cusum))
fn_logit_cusum = np.zeros(len(percentiles_logit_cusum))
for idx, percentile in enumerate(percentiles):
    for i in range(num_repeats):
        dt, _ = detect_change_in_stream_loc_batched(data_alt[i], model, window_length, percentile)
        if i % 100 == 0:
            print(f"{i} w")
        if dt > 0 and dt > true_tau_alt[i]:
            detection_delay[idx,i] = dt - true_tau_alt[i]
        if dt > 0 and dt < true_tau_alt[i]:
            fp[idx] += 1
        if dt == 0 and true_tau_alt[i] > 0:
            fn[idx] += 1
for idx, percentile in enumerate(percentiles_cusum):
    for i in range(num_repeats):
        dt_cusum, _ = li_cusum(data_alt[i], window_length, percentile)
        if i % 100 == 0:
            print(f"{i} s")
        if dt_cusum > 0 and dt_cusum > true_tau_alt[i]: #prøv and dt-true_tau_alt[i] mindre eller lik window_length
            detection_delay_cusum[idx,i] = dt_cusum - true_tau_alt[i]
        if dt_cusum > 0 and dt_cusum < true_tau_alt[i]:
            fp_cusum[idx] += 1
        if dt_cusum == 0 and true_tau_alt[i] > 0:
            fn_cusum[idx] += 1
for idx, percentile in enumerate(percentiles_logit_cusum):
    for i in range(num_repeats):
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data_alt[i], model, window_length, percentile)
        if i % 100 == 0:
            print(f"{i} l")
        if dt_logit_cusum > 0 and dt_logit_cusum > true_tau_alt[i]:
            detection_delay_logit_cusum[idx,i] = dt_logit_cusum - true_tau_alt[i]
        if dt_logit_cusum > 0 and dt_logit_cusum < true_tau_alt[i]:
            fp_logit_cusum[idx] += 1
        if dt_logit_cusum == 0 and true_tau_alt[i] > 0:
            fn_logit_cusum[idx] += 1
average_logit_cusum_delay = np.mean(detection_delay_logit_cusum,axis=1)
average_cusum_delay = np.mean(detection_delay_cusum,axis=1)
average_delay = np.mean(detection_delay,axis=1)
print(fp)
plt.figure(figsize=(15, 5))
# Plot 1: Average Detection Delay
plt.subplot(1, 3, 1)
plt.plot(false_alarm_rates, average_delay, 'o-', linewidth=2, markersize=6, color='blue')
plt.plot(false_alarm_rates, average_cusum_delay, 'o-', linewidth=2, markersize=6, color='red')
plt.plot(false_alarm_rates, average_logit_cusum_delay, 'o-', linewidth=2, markersize=6, color='green')
plt.xlim(0.8,1.0)
plt.legend(['Detection Delay NN', 'Detection Delay CUSUM', 'Detection Delay Logit CUSUM'])
plt.xlabel('Percentile Threshold Normal')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)

# Plot 2: False Positives
plt.subplot(1, 3, 2)
plt.plot(false_alarm_rates, fp, 'o-', linewidth=2, markersize=6, color='blue')
plt.plot(false_alarm_rates, fn, 'o-', linewidth=2, markersize=6, color='yellow')
plt.plot(false_alarm_rates, fp_cusum, 'o-', linewidth=2, markersize=6, color='red')
plt.plot(false_alarm_rates, fn_cusum, 'o-', linewidth=2, markersize=6, color='orange')
plt.plot(false_alarm_rates, fp_logit_cusum, 'o-', linewidth=2, markersize=6, color='purple')
plt.plot(false_alarm_rates, fn_logit_cusum, 'o-', linewidth=2, markersize=6, color='green')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Number of False Positives')
plt.title('False Positives vs Threshold')
plt.legend(['False Positives NN', 'False Negatives NN', 'False Positives CUSUM', 'False Negatives CUSUM', 'False Positives Logit CUSUM', 'False Negatives Logit CUSUM'])
plt.grid(True)

# Plot 3: Average Run Length
plt.subplot(1, 3, 3)
plt.plot(false_alarm_rates, arl, 'o-', linewidth=2, markersize=6, color='blue')
plt.plot(false_alarm_rates, arl_cusum, 'o-', linewidth=2, markersize=6, color='red')
plt.plot(false_alarm_rates, arl_logit_cusum, 'o-', linewidth=2, markersize=6, color='green')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Average Run Length')
plt.title('ARL vs Threshold')
plt.grid(True)
plt.legend(['ARL NN', 'ARL CUSUM', 'ARL Logit CUSUM'])
plt.savefig(f"thresholdsimulatednormalwologit.png")
plt.tight_layout()
plt.show()