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

from autocpd.utils import DataGenAlternative, GenDataMean, detect_change_in_stream
from sklearn.utils import shuffle

#parameters
window_length = 100
num_repeat = 20
stream_length = [200, 600, 1000]
sigma = 1
seed_val = [2023]

epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
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

predicted_labels = np.zeros(num_streams, dtype=int)
detected_cps = [None] * num_streams

for i in range(num_streams):
    stream = data_all[i]
 
    predicted_labels[i] = detect_change_in_stream(stream, model, window_length)
    print(f"Stream {i}: predicted label = {predicted_labels[i]}, true tau = {true_taus[i]}")
true_labels = y_all.flatten()

# Calculate misclassification error rate (MER)
MER = np.mean(predicted_labels != true_labels)
print(f"Misclassification Error Rate (MER): {MER:.3f}")

false_positive = np.where((predicted_labels == 1) & (true_labels == 0))
print(f"False positive: {len(false_positive[0])}")
print(false_positive)
false_negative = np.where((predicted_labels == 0) & (true_labels == 1))
print(f"False negative: {len(false_negative[0])}")











"""
example_idx = np.where(true_labels == 1)[0][0]  # Find first stream with a change
example_stream = data_all[example_idx]
pred = detect_change_in_stream(example_stream, model, window_length)
example_true_tau = true_taus[example_idx]

plt.figure(figsize=(10, 6))
plt.plot(example_stream, label="Time Series")
plt.title(f"Example Stream (Detected Change: {pred})")
plt.xlabel("Time Step")
plt.ylabel("Signal Value")
plt.axvline(x=true_taus[example_idx], color='r', linestyle='--', label="True Change")
plt.legend()
plt.grid(True)
plot_save_path = Path("datasets", "plots", f"{model_name}_binary_classification_sigma{sigma}.png")
plot_save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_save_path)
"""