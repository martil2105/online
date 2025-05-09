import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from autocpd.utils import DataGenAlternative, GenDataMean
from sklearn.utils import shuffle

#parameters
window_length = 100
num_repeat = 50
stream_length = 2000 #previous 20000
sigma = 1
seed_val = [2023]

epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 100
B_bound = np.array([0.25, 1.75])
rhos = 0
#load model
current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
#print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)
#load cusum

N_alt = num_repeat
N_null = num_repeat

#simulation
seed = 2025
#print(B_val)
detection_times = []
np.random.seed(seed)
tf.random.set_seed(seed)
# Generate null data
data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
true_tau_null = np.zeros((N_null,), dtype=np.int32)  # no change → set to 0
#generate alternative data
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
data_alt = result_alt["data"]
true_tau_alt = result_alt["tau_alt"]
#print(f"data_all: {data_all.shape}")
num_streams = data_null.shape[0] #number of streams


def mean_in_stream(stream, model, window_length):
    logits_difference = []
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1)

    logits = model.predict(windows, verbose=0)
    #print(f"logits: {logits}")
    logits_diff = logits[:,1] - logits[:,0]
    return logits_diff

def detect_change_in_stream_batched_cusum(stream, model, window_length, threshold): 
    logits_difference = []
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1)

    logits = model.predict(windows, verbose=0)
    logits_diff = logits[:,1] - logits[:,0] #d_t 
    #print(logits_diff)
    cusum_scores = []
    detection_times = 0
    S = 0
    for i in range(len(logits_diff)):
        d_t = logits_diff[i]
        S = max(0, S + d_t)
        cusum_scores.append(S)
        #print(cusum_scores)
        if S > threshold:
            detection_times = i+window_length
            break
    return detection_times, cusum_scores
alpha = 0.05
false_positives = 0
false_negative = 0
target_edd = 15
target_edd_margin = alpha * target_edd
current_edd = 0 
max_iter = 30
iter = 0
delay = []
allowed_false_positives = np.ceil(num_streams * alpha)
print(f"allowed_false_positives: {allowed_false_positives}")
threshold_high = 30
threshold_low  = 0
max_iter = 30

for iter in range(max_iter):
    threshold = 0.5 * (threshold_high + threshold_low)
    print(f"\nIter {iter+1:2d} – threshold {threshold:.4f}")

    # --- reset counters for this candidate threshold ---
    false_positives = 0
    false_negatives = 0
    delay           = []

    for i in range(num_streams):
        stream = data_alt[i]
        det_time, _ = detect_change_in_stream_batched_cusum(
                        stream, model, window_length, threshold)

        if det_time == 0:                           # no alarm
            false_negatives += 1
        elif det_time <= true_tau_alt[i]:           # alarm before change point
            false_positives += 1
        else:                                       # correct detection
            delay.append(det_time - true_tau_alt[i])

    current_edd = np.mean(delay) if delay else np.inf
    print(f"  EDD={current_edd:.2f},  FP={false_positives}")

    # -------- convergence test --------
    edd_ok = abs(current_edd - target_edd) <= target_edd_margin
    fp_ok  = false_positives <= allowed_false_positives
    if edd_ok and fp_ok:
        break

    if current_edd < target_edd or not fp_ok:
        threshold_low  = threshold      # need to be more sensitive
    else:
        threshold_high = threshold      # can be less sensitive
else:
    print("Reached max")

final_threshold = threshold

print(f"\nDone → threshold={threshold:.4f}, EDD={current_edd:.2f}, "
      f"FP={false_positives}, FN={false_negatives}")

# save the threshold
print(f"final_threshold: {final_threshold}")
output_path = Path(script_dir, "optimal_threshold_cusum_foredd20.npy")
np.save(output_path, final_threshold)
print(f"Saved {output_path}")
