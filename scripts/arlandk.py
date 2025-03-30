import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from time import time
begin_time = time()
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from autocpd.utils import DataGenAlternative, GenDataMean, detect_change_in_stream
from sklearn.utils import shuffle

#parameters
window_length = 100
num_repeat = 1
stream_length = 10000
sigma = 1.5
seed_val = [2023]
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
#k = 5
thresholds = [0.6,0.8,0.85,0.9,0.96]
k_values = [1,2,3]
#load model
current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)
#load cusum

N_alt = num_repeat
N_null = num_repeat

#simulation
seed = 2023
#print(B_val)
false_positive_count = []
MER_count = []

np.random.seed(seed)
tf.random.set_seed(seed)
result_alt = DataGenAlternative(
    N_sub=N_alt,
    B=B_val,
    mu_L=mu_L,
    n=stream_length,
    ARcoef=rhos,
    tau_bound=tau_bound,
    B_bound=B_bound,
    ar_model="Gaussian",
    sigma=sigma
)
data_alt = result_alt["data"]  # .shape (N_alt, stream_length)
true_tau_alt = result_alt["tau_alt"]  # tau index"""

# Generate null data
data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
true_tau_null = np.zeros((N_null,), dtype=np.int32)  # no change → set to 0

# Concatenate alternative and null streams into one dataset
#data_all = np.concatenate((data_alt, data_null), axis=0)  # shape: (2*num_repeat, stream_length)

# Create labels: 1 for alternative (change present), 0 for null (no change)
#y_all = np.repeat((1, 0), (N_alt, N_null)).reshape((2 * num_repeat, 1))
# Also combine true change-point locations for later reference
true_taus = np.concatenate((true_tau_alt, true_tau_null), axis=0)

# Shuffle the dataset (and ensure labels and true change-points are shuffled together)
#data_all, y_all, true_taus = shuffle(data_all, y_all, true_taus, random_state=42)

#print(true_taus)
#num_streams = data_all.shape[0]  # number of streams
num_streams = data_alt.shape[0]
false_positive_count = 0
false_negative_count = 0
true_positive_count = 0
true_negative_count = 0

def detect_change_in_stream_loc(stream, model, window_length, k,threshold, runs):
    num_windows = len(stream) - window_length + 1
    detection_time = 0
    consecutive_changes = 0
    for i in range(num_windows):
        if i % 10000 == 0:
            print(f"Current step: {i}")
        window_data = stream[i: i + window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        prob = tf.nn.softmax(logits).numpy()
        
        #print(i,prob,threshold,runs)

        if prob[0][1] > threshold: 
            consecutive_changes += 1
            if consecutive_changes == k:
                detection_time = i + window_length
                break
        else:
            consecutive_changes = 0
    return detection_time

def calculate_detection_delay(detected_time, true_tau):
    """Calculate the detection delay."""
    return max(0, detected_time - true_tau)

def process_streams(data, true_taus, model, window_length, thresholds, k):
    """Process streams to calculate detection delays and run lengths."""
    num_streams = data.shape[0]
    detection_delay = np.zeros((num_streams, len(thresholds)))
    run_length = np.zeros((num_streams, len(thresholds)))

    # Initialize counters
    false_positive_count = 0
    false_negative_count = 0
    true_positive_count = 0
    true_negative_count = 0

    for i in range(num_streams):
        stream = data[i]
        detected_time = detect_change_in_stream_loc(stream, model, window_length, k=k, threshold=threshold, runs=i)
        detection_delay[i, j] = calculate_detection_delay(detected_time, true_taus[i])
        run_length[i, j] = detected_time if detected_time > 0 else stream_length

        # Update counters
        if true_taus[i] > 0:
            if detected_time < true_taus[i]:
                false_positive_count += 1
                
            elif detected_time == 0:
                false_negative_count += 1
            elif detected_time > true_taus[i]:
                true_positive_count += 1
        elif true_taus[i] == 0:
            if detected_time > 0:
                false_positive_count += 1
                
            elif detected_time == 0:
                true_negative_count += 1

        print(f"Stream {i}, Threshold {threshold}: Detected Time = {detected_time}, True Tau = {true_taus[i]}")

    return detection_delay, run_length, false_positive_count, false_negative_count, true_positive_count, true_negative_count

average_run_length = np.zeros((len(thresholds), len(k_values)))
detection_delay = np.zeros((len(thresholds), len(k_values)))
# Process alternative and null streams
for i, threshold in enumerate(thresholds):
    for j, k in enumerate(k_values):
        detection_delay_alt, run_length_alt, fp_alt, fn_alt, tp_alt, tn_alt = process_streams(data_alt, true_tau_alt, model, window_length, thresholds, k)
        detection_delay_null, run_length_null, fp_null, fn_null, tp_null, tn_null = process_streams(data_null, true_tau_null, model, window_length, thresholds, k)
        detection_delay[i, j] = np.mean(detection_delay_alt, axis=0)
        average_run_length[i, j] = np.mean(run_length_null, axis=0)

# Combine results
#fp = fp_alt + fp_null
#fn = fn_alt + fn_null
#tp = tp_alt + tp_null
#tn = tn_alt + tn_null
#detection_delay = detection_delay_alt 

run_length = run_length_null

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
for i, threshold in enumerate(thresholds):
    plt.plot(k_values, average_run_length[i], marker="o", label=f"Threshold {threshold}")
plt.xscale("log")
plt.xlabel("k")
plt.ylabel("Average Run Length")
plt.legend()
for i, threshold in enumerate(thresholds):
    plt.subplot(1, 2, 2)
    plt.plot(k_values, detection_delay[i], marker="o", label=f"Threshold {threshold}")
plt.xscale("log")
plt.xlabel("k")
plt.ylabel("Detection Delay")
plt.legend()
plt.show()