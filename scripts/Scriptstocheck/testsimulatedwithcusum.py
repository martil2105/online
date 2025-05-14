import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from time import time
begin_time = time()

# Project path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)

from autocpd.utils import *
from sklearn.utils import shuffle

# Parameters
window_length = 100
num_repeats = 1000
stream_length = 2000
sigma = 1
seed = 2023
epsilon = 0.05
B_val = np.sqrt(8 * np.log(window_length / epsilon) / window_length)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.25, 1.75])
rhos = 0
thresholds = [99999]

# Load model
current_file = "train"
model_name = "n100N400m24l5" #modell navn fra tensorboard, "vindu,antall serier trent på, bredde på NN, layers"
logdir = Path("tensorboard_logs", f"{current_file}") 
model_path = Path(logdir, model_name, "model.keras") #mappen
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path) #modellen

np.random.seed(seed)
tf.random.set_seed(seed)

# Batched version of detection function
def detect_change_in_stream_loc_batched(stream, model, window_length, threshold):
    num_windows = len(stream) - window_length + 1 #antall vinduer
    windows = np.array([stream[i:i+window_length] for i in range(num_windows)]) #har alle vinduene i en array for raskere inferens, men dette er akkurat det samme som en while loop
    windows = np.expand_dims(windows, axis=-1)  #slik at det passer NN
    logits = model.predict(windows, verbose=0) #finner en prediction for hvert av vinduene
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1] #sannynlighet for at det finnes et changepoint i vinduet
    detection_idx = np.argmax(probs > threshold) #finner den første changepointen, hvis det ikke er noen over threshold returneres 0
    if probs[detection_idx] > threshold: #hvis sannsynligheten er større enn threshold
        detection_time = detection_idx + window_length #changepoint er siste punkt i vinduet
    else:
        detection_time = 0 #returnerer 0
    return detection_time, np.max(probs) #detection tid og den største sannsynlighete for alle vinduene



# Generate null data and compute threshold percentile
data_null = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma) #null data
num_streams = data_null.shape[0]
max_probabilities = []
max_probabilities_cusum = []
max_probabilities_logit_cusum = []
for i in range(num_streams):
    dt, max_prob = detect_change_in_stream_loc_batched(data_null[i], model, window_length, thresholds[0]) #med urealistisk høy threshold for å finne maksen under null
    max_probabilities.append(max_prob) #legger til en høyeste sannsynlighet
    dt_cusum, max_prob_cusum = li_cusum(data_null[i], window_length, thresholds[0]) #li cusum for alle vinduene med urealistisk høy threshold for å finne maksen under null
    max_probabilities_cusum.append(max_prob_cusum) #legger til i cusum listen
    dt_logit_cusum, max_prob_logit_cusum = detect_change_in_stream_batched_cusum(data_null[i], model, window_length, thresholds[0]) #logit differanse cusum
    max_probabilities_logit_cusum.append(max_prob_logit_cusum) #legges i riktig liste
    print(f"i: {i}")
false_alarm_rates = [0.8,0.85, 0.90,0.95,0.99] #forskjellige nivåer
#regular
percentile_80 = np.percentile(max_probabilities, 80) #persentil for ulike nivåer
percentile_85 = np.percentile(max_probabilities, 85)
percentile_90 = np.percentile(max_probabilities, 90)
percentile_95 = np.percentile(max_probabilities, 95)
percentile_99 = np.percentile(max_probabilities, 99)
#logit cusum
percentile_80_logit_cusum = np.percentile(max_probabilities_logit_cusum, 80)
percentile_85_logit_cusum = np.percentile(max_probabilities_logit_cusum, 85)
percentile_90_logit_cusum = np.percentile(max_probabilities_logit_cusum, 90)
percentile_95_logit_cusum = np.percentile(max_probabilities_logit_cusum, 95)
percentile_99_logit_cusum = np.percentile(max_probabilities_logit_cusum, 99)
#cusum
percentile_80_cusum = np.percentile(max_probabilities_cusum, 80)
percentile_85_cusum = np.percentile(max_probabilities_cusum, 85)
percentile_90_cusum = np.percentile(max_probabilities_cusum, 90)
percentile_95_cusum = np.percentile(max_probabilities_cusum, 95)
percentile_99_cusum = np.percentile(max_probabilities_cusum, 99)
print(f"Repeats: {num_streams}, 95th percentile: {percentile_95}")
percentiles = [percentile_80,percentile_85,percentile_90, percentile_95, percentile_99] #percentiler slik at vi kan loope gjennom dem
percentiles_cusum = [percentile_80_cusum,percentile_85_cusum,percentile_90_cusum, percentile_95_cusum, percentile_99_cusum]
percentiles_logit_cusum = [percentile_80_logit_cusum,percentile_85_logit_cusum,percentile_90_logit_cusum, percentile_95_logit_cusum, percentile_99_logit_cusum]
# Estimate ARL (Average Run Length)
arl = np.zeros((len(percentiles),num_repeats)) #array for å finne ARL for ulike percentiler
arl_cusum = np.zeros((len(percentiles),num_repeats))
arl_logit_cusum = np.zeros((len(percentiles_logit_cusum),num_repeats))
data = GenDataMean(num_repeats, stream_length, cp=None, mu=(mu_L, mu_L), sigma=sigma) #generer null data
num_streams = data.shape[0] #antall serier = repeats

for idx, percentile in enumerate(percentiles): #looper gjennom alle percentilene
    for i in range(num_streams): #looper gjennom alle serier
        dt, _ = detect_change_in_stream_loc_batched(data[i], model, window_length, percentile) #finner detection tid
        if dt > 0: #hvis det er en detection, så er det feil
            arl[idx,i] = dt #arl er detection tid
        else:
            arl[idx,i] = stream_length #hvis det ikke er en detection, så er arl lik stream length
for idx, percentile in enumerate(percentiles_cusum): 
    for i in range(num_streams):
        dt_cusum, _ = li_cusum(data[i], window_length, percentile)
        if dt_cusum > 0:
            arl_cusum[idx,i] = dt_cusum
        else:
            arl_cusum[idx,i] = stream_length
for idx, percentile in enumerate(percentiles_logit_cusum):
    for i in range(num_streams):
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data[i], model, window_length, percentile)
        if dt_logit_cusum > 0:
            arl_logit_cusum[idx,i] = dt_logit_cusum
        else:
            arl_logit_cusum[idx,i] = stream_length

arl_cusum = np.mean(arl_cusum,axis=1) #finner gjennomsnittet av arl hver percentil
arl = np.mean(arl,axis=1)
arl_logit_cusum = np.mean(arl_logit_cusum,axis=1)


result_alt =  DataGenAlternative(
                N_sub=num_repeats,
                B=B_val,
                mu_L=mu_L,
                n=stream_length,
                ARcoef=rhos,
                tau_bound=tau_bound,
                B_bound=B_bound,
                ar_model="Gaussian",
                sigma=sigma) #Data med changepoint
data_alt = result_alt["data"] #tidseriene    
true_tau_alt = result_alt["tau_alt"] #changepointene

detection_delay = np.zeros((len(percentiles),num_repeats))
detection_delay_cusum = np.zeros((len(percentiles),num_repeats))
detection_delay_logit_cusum = np.zeros((len(percentiles_logit_cusum),num_repeats))
fp_cusum = np.zeros(len(percentiles))
fn_cusum = np.zeros(len(percentiles))
fp = np.zeros(len(percentiles))
fn = np.zeros(len(percentiles))
fp_logit_cusum = np.zeros(len(percentiles_logit_cusum))
fn_logit_cusum = np.zeros(len(percentiles_logit_cusum))
for idx, percentile in enumerate(percentiles): #looper gjennom alle percentilene
    for i in range(num_repeats): #looper gjennom alle serier
        dt, _ = detect_change_in_stream_loc_batched(data_alt[i], model, window_length, percentile) #finner detection tid
        if i % 100 == 0:   #bare for å se hvor langt vi er
            print(f"{i} w")
        if dt > 0 and dt > true_tau_alt[i]: #hvis det er en detection og detection tid er større enn changepoint
            detection_delay[idx,i] = dt - true_tau_alt[i] #detection delay er detection tid minus changepoint
        if dt > 0 and dt < true_tau_alt[i]: #hvis det er en detection og detection tid er mindre enn changepoint
            fp[idx] += 1                    #regnes som falsk positiv fordi det er i "null" dataen
        if dt == 0 and true_tau_alt[i] > 0: #hvis det ikke er en detection og det finnes enn changepoint
            fn[idx] += 1                    #regnes som falsk negativ fordi det ikke er en detection
for idx, percentile in enumerate(percentiles_cusum):
    for i in range(num_repeats):
        dt_cusum, _ = li_cusum(data_alt[i], window_length, percentile)
        if i % 100 == 0:
            print(f"{i} s")
        if dt_cusum > 0 and dt_cusum > true_tau_alt[i]: #prøv and dt-true_tau_alt[i] mindre eller lik window_length
            detection_delay_cusum[idx,i] = dt_cusum - true_tau_alt[i]
        if dt_cusum > 0 and dt_cusum < true_tau_alt[i]:
            fp_cusum[idx] += 1
        if dt_cusum == 0 and true_tau_alt[i] > 0:
            fn_cusum[idx] += 1
for idx, percentile in enumerate(percentiles_logit_cusum):
    for i in range(num_repeats):
        dt_logit_cusum, _ = detect_change_in_stream_batched_cusum(data_alt[i], model, window_length, percentile)
        if i % 100 == 0:
            print(f"{i} l")
        if dt_logit_cusum > 0 and dt_logit_cusum > true_tau_alt[i]:
            detection_delay_logit_cusum[idx,i] = dt_logit_cusum - true_tau_alt[i]
        if dt_logit_cusum > 0 and dt_logit_cusum < true_tau_alt[i]:
            fp_logit_cusum[idx] += 1
        if dt_logit_cusum == 0 and true_tau_alt[i] > 0:
            fn_logit_cusum[idx] += 1
average_logit_cusum_delay = np.mean(detection_delay_logit_cusum,axis=1) #finner gjennomsnittet av detection delay for logit cusum
average_cusum_delay = np.mean(detection_delay_cusum,axis=1) #finner gjennomsnittet av detection delay for cusum
average_delay = np.mean(detection_delay,axis=1) #finner gjennomsnittet av detection delay for nn
print(fp)
plt.figure(figsize=(15, 5))
# Plot 1: Average Detection Delay
plt.subplot(1, 3, 1)
plt.plot(false_alarm_rates, average_delay, 'o-', linewidth=2, markersize=6, color='blue')
plt.plot(false_alarm_rates, average_cusum_delay, 'o-', linewidth=2, markersize=6, color='red')
plt.plot(false_alarm_rates, average_logit_cusum_delay, 'o-', linewidth=2, markersize=6, color='green')
plt.xlim(0.8,1.0)
plt.legend(['Detection Delay NN', 'Detection Delay CUSUM', 'Detection Delay Logit CUSUM'])
plt.xlabel('Percentile Threshold Normal')
plt.ylabel('Average Detection Delay')
plt.title('Detection Delay vs Threshold')
plt.grid(True)

# Plot 2: False Positives
plt.subplot(1, 3, 2)
plt.plot(false_alarm_rates, fp, 'o-', linewidth=2, markersize=6, color='blue')
plt.plot(false_alarm_rates, fn, 'o-', linewidth=2, markersize=6, color='yellow')
plt.plot(false_alarm_rates, fp_cusum, 'o-', linewidth=2, markersize=6, color='red')
plt.plot(false_alarm_rates, fn_cusum, 'o-', linewidth=2, markersize=6, color='orange')
plt.plot(false_alarm_rates, fp_logit_cusum, 'o-', linewidth=2, markersize=6, color='purple')
plt.plot(false_alarm_rates, fn_logit_cusum, 'o-', linewidth=2, markersize=6, color='green')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Number of False Positives')
plt.title('False Positives vs Threshold')
plt.legend(['False Positives NN', 'False Negatives NN', 'False Positives CUSUM', 'False Negatives CUSUM', 'False Positives Logit CUSUM', 'False Negatives Logit CUSUM'])
plt.grid(True)

# Plot 3: Average Run Length
plt.subplot(1, 3, 3)
plt.plot(false_alarm_rates, arl, 'o-', linewidth=2, markersize=6, color='blue')
plt.plot(false_alarm_rates, arl_cusum, 'o-', linewidth=2, markersize=6, color='red')
plt.plot(false_alarm_rates, arl_logit_cusum, 'o-', linewidth=2, markersize=6, color='green')
plt.xlim(0.8,1.0)
plt.xlabel('Percentile Threshold')
plt.ylabel('Average Run Length')
plt.title('ARL vs Threshold')
plt.grid(True)
plt.legend(['ARL NN', 'ARL CUSUM', 'ARL Logit CUSUM'])
plt.savefig(f"thresholdsimulatednormalwlogit.png")
plt.tight_layout()
plt.show()