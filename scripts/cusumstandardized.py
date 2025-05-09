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
num_repeat = 1
stream_length =  400
sigma = 2


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
seed = 2022
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
    prob = tf.nn.softmax(logits).numpy()[:,1]
    return logits_diff, prob

def detect_change_in_stream_batched_cusum(stream, model, window_length, threshold,drift ):
    logits_difference = []
    num_windows = len(stream) - window_length + 1
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
    windows = np.expand_dims(windows, axis=-1)

    logits = model.predict(windows, verbose=0)
    logits_diff = logits[:,1] - logits[:,0] #d_t change (negative under null) minus null logits (positive under null)
    #print(logits_diff)
    cusum_scores = []
    detection_times = 0
    S = 0
    for i in range(len(logits_diff)):
        d_t = logits_diff[i] / drift
        S = max(0, S + d_t )
        cusum_scores.append(S)
        #print(cusum_scores)
        if S > threshold:
            detection_times = i+window_length
            break
    return detection_times, cusum_scores, logits_diff


null_logits_diff = []
standardized_logits_diff = []
null_prob = []
for i in range(num_streams):
    stream = data_alt[i]
    logits_diff, prob = mean_in_stream(stream, model, window_length)
    #print(logits_diff)
    null_logits_diff.append(logits_diff)
    standardized_logits_diff.append(logits_diff - np.mean(logits_diff) / np.std(logits_diff))
    null_prob.append(prob)
print(null_logits_diff[0])
print(np.mean(null_logits_diff[0]))
print(np.std(null_logits_diff[0]))
print(np.mean(standardized_logits_diff[0]))
print(np.mean(null_prob[0]))
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(null_prob[0])
plt.subplot(1,2,2)
plt.plot(null_logits_diff[0])
plt.axvline(true_tau_alt[0]-100, color='r', linestyle='--')
plt.show()