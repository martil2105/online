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
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from autocpd.utils import DataGenAlternative, GenDataMean


tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)


window_length = 100
num_repeat = 100
stream_lengths = [1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000]
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
alpha = 0.05


current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", current_file)
model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)


def detect_change_in_stream_batched(stream, model, window_length, k, threshold):
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
                return 1, [i + window_length]
        else:
            consecutive = 0
    return 0, []


N_alt = num_repeat
N_null = num_repeat

Average_run_length = []
detection_delay = []

for stream_length in stream_lengths:
    print(f"\nTesting stream length: {stream_length}")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Generate data
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

    data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
    true_tau_null = np.zeros((N_null,), dtype=np.int32)

    # Combine and shuffle
    data_all = np.concatenate((data_alt, data_null), axis=0)
    y_all = np.repeat((1, 0), (N_alt, N_null)).reshape((-1, 1))
    true_taus = np.concatenate((true_tau_alt, true_tau_null), axis=0)
    data_all, y_all, true_taus = shuffle(data_all, y_all, true_taus, random_state=42)

    # Bonferroni threshold
    num_windows = stream_length - window_length + 1
    bonferroni_threshold = 1 - alpha / num_windows
    print(bonferroni_threshold)
    num_streams = data_all.shape[0]
    detection_times_delay = np.zeros(num_streams)
    run_length = np.zeros(num_streams)

    for i in range(num_streams):
        stream = data_all[i]
        detected, locs = detect_change_in_stream_batched(
            stream, model, window_length, k=1, threshold=bonferroni_threshold
        )

        if not locs:
            run_length[i] = stream_length
            detection_times_delay[i] = 0
        else:
            run_length[i] = locs[0]
            detection_times_delay[i] = locs[0] - true_taus[i]

        if i % 20 == 0:
            print(f"Processed stream {i}/{num_streams}")

    arl = np.mean(run_length)
    delay = np.mean(detection_times_delay)
    Average_run_length.append(arl)
    detection_delay.append(delay)

    print(f"ARL₀: {arl:.2f} | Detection Delay: {delay:.2f}")

# -----------------
# Plotting
# -----------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(stream_lengths, Average_run_length, marker='o')
plt.xlabel("Stream Length")
plt.ylabel("Average Run Length (ARL₀)")
plt.title("ARL₀ vs Stream Length")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(stream_lengths, detection_delay, marker='o', color='red')
plt.xlabel("Stream Length")
plt.ylabel("Detection Delay")
plt.title("Detection Delay vs Stream Length")
plt.grid(True)

plt.tight_layout()
#plt.savefig("arl_vs_delay.png")
plt.show()

end_time = time()
print(f"\nTotal time: {end_time - begin_time:.2f} seconds")
