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
num_repeat = 2
stream_length = 600
sigma = 1
seed_val = [2022]

epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 200
B_bound = np.array([0.25, 1.75])
rhos = 0
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
detection_times = []
np.random.seed(seed)
tf.random.set_seed(seed)
def detect_change_in_stream(stream, model, window_length,k, threshold ):
    logit_stream = []
    probs_stream = []
    num_windows = len(stream) - window_length + 1
    predictions = np.zeros(num_windows)
    consecutive_changes = 0
    detected_change = 0
    detected_change_windows = []
    for i in range(num_windows):
        window_data = stream[i: i + window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        logit_stream.append(logits[0,1]-logits[0,0])
        print(f"window {i}")
        print(f"logits: {logits}")
        print(f"difference: {logits[0,0]-logits[0,1]}")
        probs = tf.nn.softmax(logits, axis=1).numpy()#
        probs_stream.append(probs[0,1])
        #print(f"probs: {probs}")
        if probs[0,1] > threshold:
            consecutive_changes += 1
        else:
            consecutive_changes = 0
        if consecutive_changes >= k:
            detected_change = 1
            detected_change_windows.append(i+window_length)
            print(f"Change detected at window {i+window_length}")
            detection_times.append(i+window_length)
            break
    return detected_change, logit_stream, probs_stream
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
data_alt = result_alt["data"]  # .shape (N_alt, stream_length)
true_tau_alt = result_alt["tau_alt"]  # tau index"""

    # Generate null data
data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
true_tau_null = np.zeros((N_null,), dtype=np.int32)  # no change → set to 0

#print(f"data_all: {data_all.shape}")
num_streams = data_null.shape[0] #number of streams

predicted_labels = np.zeros(num_streams, dtype=int)
detected_cps = [None] * num_streams
logit_streams_null = []
probs_streams_null = []
logit_streams_alt = []
probs_streams_alt = []
for i in range(num_streams):
    stream = data_null[i]
    predicted_labels[i], logit_stream, probs_stream = detect_change_in_stream(stream, model, window_length,k=1, threshold=0.9)
    logit_streams_null.append((logit_stream))
    probs_streams_null.append((probs_stream))
    stream = data_alt[i]
    predicted_labels[i], logit_stream, probs_stream = detect_change_in_stream(stream, model, window_length,k=1, threshold=1)
    logit_streams_alt.append((logit_stream))
    probs_streams_alt.append((probs_stream))
    print(f"Stream {i}: predicted label = {predicted_labels[i]}")
print(f"average logit stream null: {np.mean(logit_streams_null, axis=1)}")
print(f"average logit stream alt: {np.mean(logit_streams_alt, axis=1)}")
print(f"average probs stream null: {np.mean(probs_streams_null, axis=1)}")
print(f"average probs stream alt: {np.mean(probs_streams_alt, axis=1)}")
plt.figure(figsize=(20, 8))
plt.subplot(2,2,1)
plt.plot(logit_streams_null[0])
plt.title("Null stream")
plt.subplot(2,2,2)
plt.plot(logit_streams_alt[1])
plt.title("Alternative stream")
plt.axvline(x=true_tau_alt[1]-window_length, color='r', linestyle='--')
plt.subplot(2,2,3)
plt.plot(probs_streams_null[0])
plt.title("Null stream with probability")
plt.subplot(2,2,4)
plt.plot(probs_streams_alt[1])
plt.title("Alternative stream with probability")
plt.axvline(x=true_tau_alt[1]-window_length, color='r', linestyle='--')
plt.savefig("newlogitanalysis.png")
plt.show()

