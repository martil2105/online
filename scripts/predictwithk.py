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
# Parameters
window_length = 100
num_repeat = 12
stream_lengths = [200, 600, 1000, 1500]  # Extended for better visualization
k_values = [1, 3, 6]
sigma = 1
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0

# Load model
current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Initialize MER matrix with NaN values for clarity
mer = np.empty((len(stream_lengths), len(k_values)))
mer[:] = np.nan  # Ensures uninitialized values are not mistaken for valid results

# Set random seed
seed = 2023
np.random.seed(seed)
tf.random.set_seed(seed)

print(f"B_val: {B_val}")

for stream_idx, stream_length in enumerate(stream_lengths):
    
    # Generate alternative and null data
    result_alt = DataGenAlternative(
        N_sub=num_repeat,
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

    data_null = GenDataMean(num_repeat, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
    true_tau_null = np.zeros((num_repeat,), dtype=np.int32)  # No change → set to 0

    # Concatenate alternative and null streams into one dataset
    data_all = np.concatenate((data_alt, data_null), axis=0)  # shape: (2*num_repeat, stream_length)

    # Create labels: 1 for alternative (change present), 0 for null (no change)
    y_all = np.repeat((1, 0), (num_repeat, num_repeat)).reshape((2 * num_repeat, 1))


    # Also combine true change-point locations
    true_taus = np.concatenate((true_tau_alt, true_tau_null), axis=0)

    # Shuffle the dataset
    data_all, y_all, true_taus = shuffle(data_all, y_all, true_taus, random_state=42)
    true_labels = y_all.flatten()  # Move this outside the loop
    print(f"Processing stream length {stream_length}: {data_all.shape}")
    num_streams = data_all.shape[0]  # Number of streams

    # Store predictions for each stream and k
    predicted_labels = np.zeros((num_streams, len(k_values)), dtype=int)

    for i in range(num_streams):
        stream = data_all[i]
        for k_idx, k in enumerate(k_values):
            predicted_labels[i, k_idx] = detect_change_in_stream(stream, model, window_length, k)

    # Compute MER for each k
    for k_idx, k in enumerate(k_values):
        MER = np.mean(predicted_labels[:, k_idx] != true_labels)
        mer[stream_idx, k_idx] = MER  # Store in MER matrix

        print(f"Stream length {stream_length}, k={k}: MER = {MER:.3f}")

#Plotting MER for Different Stream Lengths and k-values
plt.figure(figsize=(7, 6))

for k_idx, k in enumerate(k_values):
    plt.plot(stream_lengths, mer[:, k_idx], marker="o", label=f"k={k}")

plt.xlabel("Stream Length")
plt.ylabel("Misclassification Error Rate (MER)")
plt.title("MER Across Different Stream Lengths and k-values")
plt.xscale("log")  # Log scale for better visualization of long streams
plt.legend()
plt.grid(True)
plt.show()










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