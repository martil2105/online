import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from sklearn.utils import shuffle

begin_time = time()
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)

from autocpd.utils import DataGenAlternative, GenDataMean



window_length = 100
num_repeat = 100
stream_lengths =[1000, 2000, 3000, 5000,10000,15000,20000]
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
alpha = 0.2  # Overall significance level


current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)


def detect_change_batched(stream, model, window_length,  threshold):
    num_windows = len(stream) - window_length + 1
    i=0
    while i < num_windows:
        window_data = stream[i:i+window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        change_probs = probs[:, 1]
        print(i,change_probs,threshold)
        if change_probs > threshold:
            return i + window_length
        else:
            return 0

np.random.seed(seed)
tf.random.set_seed(seed)
N_null = num_repeat
N_alt = num_repeat
Average_run_length = np.zeros(len(stream_lengths))
detection_delay = np.zeros(len(stream_lengths))
false_positive = np.zeros(len(stream_lengths))
false_negative = np.zeros(len(stream_lengths))
detected = np.zeros(len(stream_lengths))
for idx, stream_length in enumerate(stream_lengths):
    print(f"\nTesting stream length: {stream_length}")

    # Generate null data
    data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)

    # Bonferroni threshold
    num_windows = stream_length - window_length + 1
    bonferroni_threshold = 1 - alpha / num_windows

    run_lengths = []

    for i in range(num_repeat):
        detection_time = detect_change_batched(
            data_null[i], model, window_length,  threshold=bonferroni_threshold
        )
        run_lengths.append(detection_time if detection_time > 0 else stream_length)

        if i % 20 == 0:
            print(f"Processed stream {i}/{N_null}")

    arl0 = np.mean(run_lengths,axis=0)
    Average_run_length[idx] = arl0 #riktig
    
    print(f"ARL for stream length {stream_length}: {arl0}")

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
    for i in range(num_repeat):
        
        detection_time = detect_change_batched(
            data_alt[i], model, window_length,  threshold=bonferroni_threshold
        )
        print(f"detection_time: {detection_time}, true_tau_alt: {true_tau_alt[i]}")
        if detection_time > 0 and detection_time > true_tau_alt[i]:
            detection_delay[idx] += detection_time - true_tau_alt[i]
            print(f"detection_delay: {detection_delay[idx]} ")
            detected[idx] += 1
        if detection_time > 0 and detection_time < true_tau_alt[i]:
            false_positive[idx]+=1
        if detection_time == 0 and true_tau_alt[i] > 0:
            false_negative[idx]+=1
    if detected[idx] > 0:
        detection_delay[idx] = detection_delay[idx] / detected[idx]
print(f"detection_delay: {detection_delay}, detected: {detected}")
print(f"false_positive: {false_positive}")
print(f"false_negative: {false_negative}")



plt.figure(figsize=(12, 6))
# Plot 1: Average Detection Delay
plt.subplot(1, 3, 1)
plt.plot(stream_lengths, detection_delay, 'o-', linewidth=2, markersize=5, color='blue')
plt.xlabel('Stream Length (Threshold = 1-alpha/(num_windows))')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)

# Plot 2: False Positives
plt.subplot(1, 3, 2)
plt.plot(stream_lengths, false_positive, 'o-', linewidth=2, markersize=5, color='blue')
plt.plot(stream_lengths, false_negative, 'o-', linewidth=2, markersize=5, color='red')
plt.xlabel('Stream Length (Threshold = 1-alpha/(num_windows))')
plt.ylabel('Number of False Positives')
plt.title('False Positives vs Threshold')
plt.legend(['False Positives', 'False Negatives'])
plt.grid(True)

# Plot 3: Average Run Length
plt.subplot(1, 3, 3)
plt.plot(stream_lengths, Average_run_length, 'o-', linewidth=2, markersize=5, color='blue')
plt.xlabel('Stream Length (Threshold = 1-alpha/num_windows)')
plt.ylabel('Average Run Length')
plt.title('ARL vs Threshold')
plt.grid(True)
plt.savefig(f"bonferronithreshold.png")
plt.tight_layout()
plt.show()
end_time = time()
print(f"\nTotal time: {end_time - begin_time:.2f} seconds")
