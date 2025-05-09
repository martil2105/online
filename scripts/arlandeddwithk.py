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
num_repeat = 200 #200
stream_length = 2000
sigma = 1
seed_val = [2023]
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
#k = 5
thresholds = [0.6,0.7,0.8,0.85,0.9,0.95,0.99]
k_values = [1,3,5,7]
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

data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
num_streams = data_alt.shape[0]
def detect_change_batched(stream, model, window_length, k, threshold):
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1)

    logits = model.predict(windows, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()
    change_probs = probs[:, 1]

    consecutive = 0
    for i, prob in enumerate(change_probs):
        if prob > threshold:
            consecutive += 1
            if consecutive >= k:
                return i + window_length
        else:
            consecutive = 0
    return 0

average_run_length = np.zeros((len(thresholds), len(k_values)))
detection_delay = np.zeros((len(thresholds), len(k_values)))
false_positive = np.zeros((len(thresholds), len(k_values)))
false_negative = np.zeros((len(thresholds), len(k_values)))
# Process alternative and null streams

for i, threshold in enumerate(thresholds):
    print(f"Threshold: {threshold}")
    for j, k in enumerate(k_values):
        print(f"K: {k}")
        run_lengths = []
        for l in range(num_repeat):
            detection_time = detect_change_batched(data_null[l], model, window_length, k, threshold)
            run_lengths.append(detection_time if detection_time > 0 else stream_length)
        average_run_length[i, j] = np.mean(run_lengths)
        detection_delays = []
        for l in range(num_repeat):
            detection_time = detect_change_batched(data_alt[l], model, window_length, k, threshold)
            #print(f"Detection time: {detection_time}",true_tau_alt[l])
            if detection_time > 0 and detection_time > true_tau_alt[l]:
                detection_delays.append(detection_time - true_tau_alt[l])
                print(f"Detection delay: {detection_time - true_tau_alt[l]}")
                print(f"Detection time: {detection_time}",true_tau_alt[l])
            if detection_time > 0 and detection_time < true_tau_alt[l]:
                false_positive[i, j] += 1
            if detection_time == 0 and true_tau_alt[l] > 0:
                false_negative[i, j] += 1
        detection_delay[i, j] = np.mean(detection_delays)
print(detection_delay)
print(f"Average run length: {average_run_length}")
print(f"False positive: {false_positive}")
false_positive_rate = false_positive / num_repeat
false_negative_rate = false_negative / num_repeat
expected_detection_delay = np.mean(detection_delay, axis=1)
print(f"Expected detection delay: {expected_detection_delay}")
plt.figure(figsize=(20, 8))
# Plot 1: Detection delay
plt.subplot(2, 2, 1)
for i in range(len(k_values)):
    plt.plot(thresholds, detection_delay[:,i], "--",label=f"K={k_values[i]}",marker="o", markersize=6)
plt.xlabel("Threshold")
plt.ylabel("Detection delay")
plt.title("Detection delay vs Threshold")
plt.legend()

# Plot 2: Average run length
plt.subplot(2, 2, 2)
for i in range(len(k_values)):
    plt.plot(thresholds, average_run_length[:,i], "--", label=f"K={k_values[i]}",marker="o", markersize=6)
plt.xlabel("Threshold")
plt.ylabel("Average run length")
plt.title("Average run length vs Threshold")
plt.legend()

# Plot 3: False positive
plt.subplot(2, 2, 3)
for i in range(len(k_values)):
    plt.plot(thresholds, false_positive_rate[:,i], "--", label=f"K={k_values[i]}",marker="o", markersize=6)
plt.xlabel("Threshold")
plt.ylabel("False positive rate")
plt.title("False positive rate vs Threshold")
plt.legend()

# Plot 4: False negative
plt.subplot(2, 2, 4)
for i in range(len(k_values)):
    plt.plot(thresholds, false_negative_rate[:,i], "--", label=f"K={k_values[i]}",marker="o", markersize=6)
plt.xlabel("Threshold")
plt.ylabel("False negatives")
plt.title("False negative vs Threshold")
plt.legend()
#plt.savefig(f"arlandeddwithk{stream_length}.png")
plt.tight_layout()
plt.show()
