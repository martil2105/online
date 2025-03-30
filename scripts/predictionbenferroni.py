import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from sklearn.utils import shuffle
import time
begin_time = time.time()
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from autocpd.utils import DataGenAlternative, GenDataMean
# -----------------
# Parameters
# -----------------
window_length = 100
num_repeat = 5     
stream_lengths = [1000, 2000, 3000, 5000]
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
alpha = 0.05          # Desired overall significance level

# -----------------
# Load model
# -----------------
current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)



def detect_change_in_stream_loc(stream, model, window_length, k, threshold,runs):
    """
    Slide a window over the stream and return detection time(s)
    when the softmax probability exceeds the given threshold.
    The detection is declared if 'k' consecutive windows are above threshold.
    """
    num_windows = len(stream) - window_length + 1
    consecutive_changes = 0
    detected_change_locs = []
    i = 0
    while i < num_windows:
        window_data = stream[i: i + window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        prob = tf.nn.softmax(logits).numpy()
        # Debug print: you can comment out the next line if too verbose
        print(f"Window {i}, probability: {prob}, threshold: {threshold},runs: {runs}")
    
        if prob[0][1] > threshold:
            consecutive_changes += 1
            if consecutive_changes >= k:
                # Record the detection time as the last index in this window
                detected_change_locs.append(i + window_length)
                break
        else:
            consecutive_changes = 0
        i += 1
        
    detected = 1 if len(detected_change_locs) > 0 else 0
    return detected, detected_change_locs

N_alt = num_repeat
N_null = num_repeat

Average_run_length = []  # to store the average run length for each stream length
detection_delay = [] 
for stream_length in stream_lengths:
    print(f"\nTesting stream length: {stream_length}")
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
    data_alt = result_alt["data"]  # .shape (N_alt, stream_length)

    true_tau_alt = result_alt["tau_alt"]  # tau index

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

    # Compute the number of windows and Bonferroni threshold for the current stream length.
    num_windows = stream_length - window_length + 1
    bonferroni_threshold = 1- alpha / num_windows

    num_streams = data_all.shape[0]

    detection_times_delay = np.zeros(num_streams)
    run_length = np.zeros(num_streams)
    for i in range(num_streams):
        data = data_all[i]
        detected_change, detected_change_locs = detect_change_in_stream_loc(data, model, window_length, k=1, threshold=bonferroni_threshold, runs=i)
        print(i)
        if detected_change_locs == []:
            run_length[i] = stream_length
            detection_times_delay[i] = 0
        else:
            run_length[i] = detected_change_locs[0]
            detection_times_delay[i] = detected_change_locs[0] - true_taus[i]

    # Compute the average detection time (ARL₀) for this stream length.

    arl0 = np.mean(run_length)
    Average_run_length.append(arl0)
    print(f"Average Run Length (ARL₀) for stream length {stream_length}: {arl0}")
    detection_delay.append(np.mean(detection_times_delay))
    print(f"Average detection delay: {detection_delay}")
print(f"Average detection delay: {detection_delay}")
print(f"Average run length: {Average_run_length}")
plt.figure(figsize=(12, 5))

# Plot for Average Run Length (ARL₀)
plt.subplot(1, 2, 1)
plt.plot(stream_lengths, Average_run_length, marker='o', linestyle='-')
plt.xlabel("Stream Length")
plt.ylabel("Average Run Length (ARL₀)")
plt.title("ARL₀ vs Stream Length")
plt.grid(True)

# Plot for Detection Delay
plt.subplot(1, 2, 2)
plt.plot(stream_lengths, detection_delay, marker='o', linestyle='-', color='red')
plt.xlabel("Stream Length")
plt.ylabel("Detection Delay")
plt.title("Detection Delay vs Stream Length")
plt.grid(True)
plt.legend()
plt.savefig("arl_vs_delay.png")
plt.tight_layout()
plt.show()
end_time = time.time()
print(f"Time taken: {end_time - begin_time} seconds")