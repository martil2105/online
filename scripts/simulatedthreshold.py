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
num_repeats = [1000]
stream_length = 10000
sigma = 1
seed_val = [2023]
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
#k = 5
thresholds = [0.999999999999 ]

#load model
current_file = "traincpd"
model_name = "n100N400m24l1cpd"
logdir = Path("tensorboard_logs", f"{current_file}")
model_path = Path(logdir, model_name, "model.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)

#simulation
seed = 2023
#print(B_val)

np.random.seed(seed)
tf.random.set_seed(seed)
def detect_change_in_stream_loc(stream, model, window_length, k,threshold, runs):
    num_windows = len(stream) - window_length + 1
    detection_time = 0
    consecutive_changes = 0
    probabilities = []
    for i in range(num_windows):
        window_data = stream[i: i + window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        prob = tf.nn.softmax(logits).numpy()
        probabilities.append(prob[0][1])
        #print(i,prob,threshold,runs)

        if prob[0][1] > threshold: 
            detection_time = i + window_length
            break
    highest_probability = max(probabilities)
    return detection_time , highest_probability

percentiles = []

for num_repeat in num_repeats:  # Iterate through each number of repeats
    data_null = GenDataMean(num_repeat, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma)  # Generate new null data
    num_streams = data_null.shape[0]
    max_probabilities = []  # Reset for each number of repeats

    for i in range(num_streams):
        detection_time, highest_probability = detect_change_in_stream_loc(data_null[i], model, window_length, k=1, threshold=thresholds, runs=i)
        max_probabilities.append(highest_probability)

    percentile_95 = np.percentile(max_probabilities, 95)
    percentiles.append(percentile_95)                                                              
    print(f"Repeats: {num_repeat}, 95th percentile: {percentile_95}")

print(percentiles)
arl = []
for num_repeat in num_repeats:
    data = GenDataMean(num_repeat,stream_length,cp=None,mu=(mu_L,mu_L),sigma=sigma)
    num_streams = data.shape[0]
    
    for i in range(num_streams):
        print(i)
        detection_time, highest_probability = detect_change_in_stream_loc(data[i],model,window_length,k=1,threshold=percentiles[0],runs=i)
        if detection_time == 0:
            arl.append(stream_length)
        else:
            arl.append(detection_time)

print(arl)
print(np.mean(arl))
