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
num_repeat = 1000
stream_lengths = [2000,5000]
sigma = 1
seed_val = [2022]

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


def detect_change_in_stream_cusum(stream, model, window_length, threshold): 
    logits_difference = []
    cusum_scores = np.zeros(len(stream) - window_length + 1)
    num_windows = len(stream) - window_length + 1
    i=0
    while i < num_windows:
        window_data = stream[i:i+window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        logits_diff = logits[:,1] - logits[:,0] #d_t class 1 - class 0
        detection_times = 0
        S = 0
        d_t = logits_diff
        S = max(0, S + d_t)
        cusum_scores[i] = S
        i+=1
        #print(cusum_scores)
        if S > threshold:
            detection_times = i+window_length
            break
    return detection_times, cusum_scores
dd = []
arls = []
fp = []
fn = []
std_arl0 = []
std_dd = []
for stream_length in stream_lengths:
    N_alt = num_repeat
    N_null = num_repeat

    #simulation
    seed = 2023
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


    output_path = Path(script_dir, "optimal_threshold_cusum_edd20.npy")
    threshold_final = np.load(output_path)
    print(f"threshold_final: {threshold_final}")
    arl0_test = np.zeros(num_streams)
    false_positive = np.zeros(num_streams)
    false_negative = np.zeros(num_streams)
    detection_delay = np.zeros(num_streams)
    for i in range(num_streams):
        stream_null = data_null[i]
        detection_time, cusum_scores = detect_change_in_stream_cusum(stream_null, model, window_length, threshold_final)
        arl0_test[i] = stream_length if detection_time == 0 else detection_time
        if i % 5 == 0:
            print(f"Processed stream {i}/{num_streams}")
        stream_alt = data_alt[i]
        detection_time, _ = detect_change_in_stream_cusum(stream_alt, model, window_length, threshold_final)
        if detection_time > 0 and detection_time > true_tau_alt[i]:
            detection_delay[i] = detection_time - true_tau_alt[i]
            if detection_time > 0 and detection_time < true_tau_alt[i]:
                false_positive[i] += 1
            if detection_time == 0:
                false_negative[i] += 1

    arls.append(np.mean(arl0_test))
    fp.append(np.mean(false_positive))
    fn.append(np.mean(false_negative))
    dd.append(np.mean(detection_delay))
    std_arl0.append(np.std(arl0_test))
    std_dd.append(np.std(detection_delay))
print(f"standard deviation of arl0: {std_arl0}")
print(f"standard deviation of dd: {std_dd}")
print(f"false positive: {fp}")
print(f"false negative: {fn}")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(stream_lengths, arls, "-", label="ARL₀")
plt.xlabel("Stream length")
plt.ylabel("ARL₀")
plt.title("ARL₀ vs Stream length")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(stream_lengths, dd, "-", label="Detection Delay")
plt.xlabel("Stream length")
plt.ylabel("Detection Delay")
plt.title("Detection Delay vs Stream length")
plt.legend()

plt.show()
#standard deviation of arl0: [380.0537792878792, 1428.1090180483423]
#standard deviation of dd: [19.269727320333313, 18.22641486963358]
 