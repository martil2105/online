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
num_repeat = 50
stream_length = 200000
sigma = 1
seed_val = [2023]
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
#k = 5
thresholds = [0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]

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
print(B_val)
false_positive_count = []
MER_count = []

np.random.seed(seed)
tf.random.set_seed(seed)
"""result_alt = DataGenAlternative(
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
result_alt= GenDataMean(N_alt, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
data_alt = result_alt["data"]  # .shape (N_alt, stream_length)

true_tau_alt = result_alt["tau_alt"]  # tau index"""

# Generate null data
data_alt = GenDataMean(N_alt, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
true_tau_null = np.zeros((N_null,), dtype=np.int32)  # no change → set to 0

# Concatenate alternative and null streams into one dataset
data_all = np.concatenate((data_null, data_null), axis=0)  # shape: (2*num_repeat, stream_length)

# Create labels: 1 for alternative (change present), 0 for null (no change)
y_all = np.repeat((1, 0), (N_alt, N_null)).reshape((2 * num_repeat, 1))
# Also combine true change-point locations for later reference
#true_taus = np.concatenate((true_tau_alt, true_tau_null), axis=0)

# Shuffle the dataset (and ensure labels and true change-points are shuffled together)
#data_all, y_all, true_taus = shuffle(data_all, y_all, true_taus, random_state=42)
data_all, y_all = shuffle(data_all, y_all, random_state=42)
#print(true_taus)
num_streams = data_all.shape[0]  # number of streams
change_point_location = []
probability_at_change_point = []
detection_delay = []
def detect_change_in_stream_loc(stream, model, window_length, k,threshold, runs):
    num_windows = len(stream) - window_length + 1
    predictions = np.zeros(num_windows)
    detected_change = 0
    detected_change_locs = []
    consecutive_changes = 0
    i = 0
    while i < num_windows:
        window_data = stream[i: i + window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        
        prob = tf.nn.softmax(logits).numpy()
        print(i,prob,threshold,runs)
    
        #detected_change = np.argmax(prob)
        if prob[0][1] > threshold:
            consecutive_changes += 1
            #detected_change = 1
            #change_point_location.append(i+window_length-1)
            #probability_at_change_point.append(prob[0][1])

            #break
            if consecutive_changes >= k:
                detected_change = 1
                detected_change_locs.append(i + window_length)  # Last index in the window
                #i += window_length  # Skip ahead by the window length
                break 

        else:
            consecutive_changes = 0
            i += 1                 
    return detected_change, detected_change_locs

run_length = np.zeros((num_streams,len(thresholds)))
detection_delay = np.zeros((num_streams,len(thresholds)))
for i in range(num_streams):
    for j, threshold in enumerate(thresholds):
        data = data_all[i]
        detected_change, detected_change_locs = detect_change_in_stream_loc(data, model, window_length, k=1, threshold=threshold,runs=i)
        print(i)
        if detected_change_locs == []:
            run_length[i, j] = stream_length
            #detection_delay[i, j] = 0
        else:
            run_length[i, j] = detected_change_locs[0]
            #detection_delay[i, j] = detected_change_locs[0] - true_taus[i]

print(f"run_length: {run_length}")
total_run_length = 0

average_run_length = np.mean(run_length, axis=0)
print(f"average_run_length: {average_run_length}")

end_time = time()
print(f"Time taken: {end_time - begin_time}")


average_run_length = np.mean(run_length, axis=0)

std_run_length = np.std(run_length, axis=0)
medians = np.median(run_length, axis=0)
q25 = np.percentile(run_length, 25, axis=0)
q75 = np.percentile(run_length, 75, axis=0)



plt.figure(figsize=(8, 5))
plt.errorbar(thresholds, average_run_length, fmt='o-', capsize=5, yerr=std_run_length)
plt.fill_between(thresholds, q25, q75, alpha=0.3)
plt.xlabel("Threshold")
plt.ylabel("Average Run Length (ARL)")
plt.title("ARL vs Threshold (Under Null Hypothesis)")
plt.yscale("log")  # Optional
plt.grid(True)
plt.tight_layout()
plt.savefig("arl_vs_thresholdq.png")
#plt.show()

plt.figure(figsize=(8, 5))
plt.errorbar(thresholds, average_run_length, fmt='o-', capsize=5)
plt.fill_between(thresholds, q25, q75, alpha=0.3)
plt.xlabel("Threshold")
plt.ylabel("Average Run Length (ARL)")
plt.title("ARL vs Threshold (Under Null Hypothesis)")
plt.yscale("log")  # Optional
plt.grid(True)
plt.tight_layout()
plt.savefig("arl_vs_thresholdqwithoutstd.png")
plt.show()