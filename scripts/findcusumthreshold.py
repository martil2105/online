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
num_repeat = 300
stream_length = 20000
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

predicted_labels = np.zeros(num_streams, dtype=int)
detected_cps = [None] * num_streams
false_alarms = 0

logits_diffs = []
logits_diffs_max = []

"""
for i in range(num_streams):
    stream = data_alt[i]
    logits_diff = mean_in_stream(stream, model, window_length)
    logits_diffs.append(logits_diff.mean())
"""
#print(logits_diffs)
max_cusum_scores = []
for i in range(100):
    stream = data_null[i]
    detection_time, cusum_scores = detect_change_in_stream_batched_cusum(stream, model, window_length, threshold=10)
    max_cusum_scores.append(np.max(cusum_scores))
#print(f"max_cusum_scores: {max_cusum_scores}")
print(np.std(max_cusum_scores))
print(np.max(max_cusum_scores))
threshold_high = np.max(max_cusum_scores)
threshold_low = 0
target_arl0 = 2000
current_arl0 = 0
arl0 = np.zeros(num_streams)
arl0_margin = 0.0125 * target_arl0  
max_iter = 50
iter = 0
#binary search
while abs(current_arl0 - target_arl0) > arl0_margin:
    iter += 1
    if iter == max_iter:
        print(f"Threshold stalled at {threshold:.4f}. Stopping.")
        break
    threshold = (threshold_high + threshold_low) * 0.5
    print(f"threshold: {threshold:.4f}")

    for i in range(num_streams):
        stream = data_null[i]
        detection_time, cusum_scores = detect_change_in_stream_batched_cusum(stream, model, window_length, threshold)
        arl0[i] = stream_length if detection_time == 0 else detection_time
    print(f"arl0: {arl0}")
    current_arl0 = np.mean(arl0)
    diff = current_arl0 - target_arl0
    print(f"current ARL₀: {current_arl0:.2f}  (diff = {diff:.2f})")

    if current_arl0 < target_arl0:
        threshold_low = threshold
        print(f"  raising threshold_low → {threshold_low:.4f}")
    else:
        
        threshold_high = threshold
        print(f"  lowering threshold_high → {threshold_high:.4f}")

print(f"Done: final threshold = {threshold:.4f}, ARL₀ ≈ {current_arl0-target_arl0} (target ± {arl0_margin:.2f})")
output_path = Path(script_dir, "optimal_threshold_cusum_forarl02000.npy")
np.save(output_path, threshold)
print(f"Saved optimal threshold to {output_path}")
threshold_final = np.load(output_path)
#2.5595
data_null_test = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
arl0_test = np.zeros(num_streams)
for i in range(num_streams):
    stream = data_null_test[i]
    detection_time, cusum_scores = detect_change_in_stream_batched_cusum(stream, model, window_length, threshold_final)
    arl0_test[i] = stream_length if detection_time == 0 else detection_time
    if i % 20 == 0:
        print(f"Processed stream {i}/{num_streams}")

test_mean_arl0 = np.mean(arl0_test)
test_std_arl0 = np.std(arl0_test)
print(f"\nTest Result: Mean ARL₀ = {test_mean_arl0:.2f}")
print(f"Test Result: Std Dev ARL₀ = {test_std_arl0:.2f}")
print(np.min(arl0_test))
print(np.max(arl0_test))
alpha = 0.05 # for 95% confidence
z_score = 1.96 # Z-score for 95% confidence
ci_margin = z_score * (test_std_arl0 / np.sqrt(num_streams))
print(f"95% CI for Test Mean ARL₀: [{test_mean_arl0 - ci_margin:.2f}, {test_mean_arl0 + ci_margin:.2f}]")