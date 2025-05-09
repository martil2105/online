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
num_repeat = 200 #200
stream_length = 5000
sigma = 1
seed_val = [2023]
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
#k = 5
thresholds = [0.6,0.7,0.8,0.85,0.9,0.95]
k_values = [1,3,5,7]
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
#print(B_val)
false_positive_count = []
MER_count = []

np.random.seed(seed)
tf.random.set_seed(seed)
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

data_null = GenDataMean(N_null, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)
num_streams = data_alt.shape[0]
def detect_change_batched(stream, model, window_length, threshold):
    detection_time = 0  
    count = 0
    while detection_time == 0:
        count += 1
        num_windows = len(stream) - window_length + 1
        windows = np.array([stream[i:i+window_length] for i in range(num_windows)])
        windows = np.expand_dims(windows, axis=-1)
        print(windows.shape)
        logits = model.predict(windows, verbose=0)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        change_probs = probs[:, 1]

        consecutive = 0
        for i, prob in enumerate(change_probs):
            if prob > threshold:
                detection_time = i + window_length
                break
            else:
                0
        stream = np.concatenate((stream, GenDataMean(1, 2*stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)[0]), axis=0)
    return detection_time, count

detection, count  = detect_change_batched(data_null[0], model, window_length, 0.95)
print(detection, count)
