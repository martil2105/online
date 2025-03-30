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
stream_length = 200
sigma = 1
seed_val = [2023]

epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
k = 5
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
for stream_length in [1500]:
        
    np.random.seed(seed)
    tf.random.set_seed(seed)
    result_alt =  DataGenAlternative(
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
    data_alt = result_alt["data"]      # .shape (N_alt, stream_length)

    true_tau_alt = result_alt["tau_alt"] # tau index

        # Generate null data
    data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
    true_tau_null = np.zeros((N_null,), dtype=np.int32)  # no change → set to 0

        # Concatenate alternative and null streams into one dataset
    data_all = np.concatenate((data_alt, data_null), axis=0)  # shape: (2*num_repeat, stream_length)

        # Create labels: 1 for alternative (change present), 0 for null (no change)
    y_all = np.repeat((1, 0), (N_alt, N_null)).reshape((2 * num_repeat, 1))
    # Also combine true change-point locations for later reference
    true_taus = np.concatenate((true_tau_alt, true_tau_null), axis=0)

    # Shuffle the dataset (and ensure labels and true change-points are shuffled together)
    data_all, y_all, true_taus = shuffle(data_all, y_all, true_taus, random_state=42)
    print(f"data_all: {data_all.shape}")
    num_streams = data_all.shape[0] #number of streams

    def detect_change_in_stream_loc(stream, model, window_length, k=k):
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
            
            output = np.argmax(logits, axis=1)[0]
            predictions[i] = output  # Save the binary prediction
        
            # Update consecutive count based on the output
            if output == 1:
                consecutive_changes += 1
            else:
                consecutive_changes = 0

            # If the count reaches the threshold, register a detection.
            if consecutive_changes >= k:
                detected_change = 1
                detected_change_locs.append(i + window_length - 1)  # Last index in the window
                i += window_length  # Skip ahead by the window length
                consecutive_changes = 0  # Reset count after detection
            else:
                i += 1  
                
        return detected_change, detected_change_locs
    true_labels = y_all.flatten()
    predicted_labels = np.zeros(num_streams, dtype=int)
    detected_cps = [None] * num_streams
    false_positive = 0
    delay = []
    for i in range(num_streams):
        stream = data_all[i]
        detected_change, detected_locs = detect_change_in_stream_loc(stream, model, window_length,k)
        predicted_labels[i] = detected_change  # Binary 
        detected_cps[i] = detected_locs  
        if true_taus[i] == 1: #if there is a true changepoint
            for cp in detected_locs: 
                if cp < true_taus[i]: #if the changepoint is before its a false positive
                    false_positive += 1
                else:
                    delay.append(cp - true_taus[i]) #if the changepoint is after its a delay
                    break
        else: #if there is no true changepoint, all is false positive
            if len(detected_locs) > 0:
                false_positive += 1
                
        print(f"Stream {i}: predicted label = {predicted_labels[i]}, detected change-points = {detected_cps[i]}, true tau = {true_taus[i]}")
    false_positive_count.append(false_positive)
    # Calculate misclassification error rate (MER)
    MER = np.mean(predicted_labels != true_labels)
    MER_count.append(MER)
    print(f"Misclassification Error Rate (MER): {MER:.3f}")
    print(predicted_labels, detected_cps, true_taus)
"""
plt.figure(figsize=(10, 6))
plt.plot([400, 800, 1600, 2000], false_positive_count, marker='o', linestyle='-')
plt.title(f'False Positives vs. Stream Lengths for k={k}')
plt.xlabel('Stream Length')
plt.ylabel('Number of False Positives')
plt.grid(True)
plt.show()"""
end_time = time()
print(f"Time taken: {end_time - begin_time} seconds")