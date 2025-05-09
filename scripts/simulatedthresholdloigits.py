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

from autocpd.utils import DataGenAlternative, GenDataMean, detect_change_in_stream
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
thresholds = [-1000]

# Load model
current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)

np.random.seed(seed)
tf.random.set_seed(seed)

# Batched version of detection function
def detect_change_in_stream_loc_batched(stream, model, window_length, threshold):
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1) 
    logits = model.predict(windows, verbose=0)
    logits_diff = logits[:,1] - logits[:,0]
    max_logit = np.max(logits_diff)
    detection_idx = np.argmax(logits_diff>threshold)
    print(f"detection_idx: logits_diff[detection_idx]: {logits_diff[detection_idx]}, threshold: {threshold}")
    if logits_diff[detection_idx] > threshold:
        #print(f"logits_diff[detection_idx]: {logits_diff[detection_idx]}, threshold: {threshold}")
        detection_time = detection_idx + window_length
    else:
        detection_time = 0
    return detection_time, max_logit, logits_diff

# Generate null data and compute threshold percentile
data_null = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
num_streams = data_null.shape[0]
max_probabilities = []

for i in range(num_streams):
    dt, max_prob, _ = detect_change_in_stream_loc_batched(data_null[i], model, window_length, thresholds[0])
    max_probabilities.append(max_prob)
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99]
percentile_80 = np.percentile(max_probabilities, 80)
percentile_85 = np.percentile(max_probabilities, 85)
percentile_95 = np.percentile(max_probabilities, 95)
percentile_99 = np.percentile(max_probabilities, 99)
percentile_90 = np.percentile(max_probabilities, 90)
print(f"Repeats: {num_streams}, 95th percentile: {percentile_95}")
percentiles = [percentile_80,percentile_85,percentile_90, percentile_95, percentile_99]
# Estimate ARL (Average Run Length)
arl = np.zeros((len(percentiles),num_repeats))
data = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=1)
num_streams = data.shape[0]
for idx, percentile in enumerate(percentiles):
    for i in range(num_streams):
        dt, _ ,_= detect_change_in_stream_loc_batched(data[i], model, window_length, percentile)
        if dt > 0:
            arl[idx,i] = dt
        else:
            arl[idx,i] = stream_length
            
print(f"arl: {arl}")
arl = np.mean(arl,axis=1)
print(f"arl mean: {arl}")
print(np.std(arl))
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
fp = np.zeros(len(percentiles))
fn = np.zeros(len(percentiles))
for idx, percentile in enumerate(percentiles):
    for i in range(num_repeats):
        dt, _ ,_= detect_change_in_stream_loc_batched(data_alt[i], model, window_length, percentile)
        #print(f"dt: {dt}, true_tau_alt[i]: {true_tau_alt[i]} {i}")
        if dt > 0 and dt > true_tau_alt[i]:
            detection_delay[idx,i] = dt - true_tau_alt[i]
        if dt > 0 and dt < true_tau_alt[i]:
            fp[idx] += 1
        if dt == 0 and true_tau_alt[i] > 0:
            fn[idx] += 1

print(f"detection_delay: {detection_delay}")
print(f"detection_delay shape: {detection_delay.shape}")
print(f"detection_delay mean: {np.mean(detection_delay,axis=1)}")
#print(detection_delay)
print(arl)
print(np.mean(detection_delay,axis=1))
average_delay = np.mean(detection_delay,axis=1)
print(fp)
plt.figure(figsize=(15, 5))

# Plot 1: Average Detection Delay
plt.subplot(1, 3, 1)
plt.plot(false_alarm_rates, average_delay, 'o-', linewidth=2, markersize=6)
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)

# Plot 2: False Positives
plt.subplot(1, 3, 2)
plt.plot(false_alarm_rates, fp, 'o-', linewidth=2, markersize=6, color='blue')
plt.plot(false_alarm_rates, fn, 'o-', linewidth=2, markersize=6, color='red')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Number of False Positives')
plt.title('False Positives vs Threshold')
plt.legend(['False Positives', 'False Negatives'])
plt.grid(True)

# Plot 3: Average Run Length
plt.subplot(1, 3, 3)
plt.plot(false_alarm_rates, arl, 'o-', linewidth=2, markersize=6)
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Average Run Length')
plt.title('ARL vs Threshold')
plt.grid(True)
plt.savefig(f"thresholdsimulated.png")
plt.tight_layout()
plt.show()