import os
import posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def GenDataMean(N, n, cp, mu, sigma):
    """
    The function  generates the data for change in mean with Gaussian noise.
    When "cp" is None, it generates the data without change point.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    mu : float
        the piecewise mean
    sigma : float
        the standard deviation of Gaussian distribution

    Returns
    -------
    numpy array
        2D array with size (N, n)
    """
    if cp is None:
        data = np.random.normal(mu[0], sigma, (N, n))
    else:
        data1 = np.random.normal(mu[0], sigma, (N, cp))
        data2 = np.random.normal(mu[1], sigma, (N, n - cp))
        data = np.concatenate((data1, data2), axis=1)
    return data

def DataGenAlternative(
    N_sub,
    B,
    mu_L,
    n,
    B_bound,
    sigma,
    ARcoef=0.0,
    tau_bound=2,
    ar_model="Gaussian",
    scale=0.1,
    
):
    """
    This function genearates the simulation data from alternative model of change in mean.

    Parameters
    ----------
    N_sub : int
        The sample size of simulation data.
    B : float
        The signal-to-noise ratio of parameter space.
    mu_L : float
        The single at the left of change point.
    n : int
        The length of time series.
    B_bound : list, optional
        The upper and lower bound scalars of signal-to-noise.
    ARcoef : float, optional
        The autoregressive parameter of AR(1) model, by default 0.0
    tau_bound : int, optional
        The lower bound of change point, by default 2
    ar_model : str, optional
        The different models, by default 'Gaussian'. ar_model="AR0" means AR(1)
        noise with autoregressive parameter 'ARcoef'; ar_model="ARH" means
        Cauchy noise with scale parameter 'scale'; ar_model="ARrho" means AR(1)
        noise with random autoregressive parameter 'scale';
    scale : float, optional
        The scale parameter of Cauchy distribution, by default 0.1
    sigma : float, optional
        The standard variance of normal distribution, by default 1.0

    Returns
    -------
    dict
        data: size (N_sub,n);
        tau_alt: size (N_sub,); the change points
        mu_R: size (N_sub,); the single at the right of change point
    """
    tau_all = np.random.randint(low=tau_bound, high=n - tau_bound, size=N_sub)
    eta_all = tau_all / n
    mu_R_abs_lower = B / np.sqrt(eta_all * (1 - eta_all))
    # max_mu_R = np.max(mu_R_abs_lower)
    sign_all = np.random.choice([-1, 1], size=N_sub)
    mu_R_all = np.zeros((N_sub,))
    data = np.zeros((N_sub, n))
    for i in range(N_sub):
        mu_R = mu_L - sign_all[i] * np.random.uniform(
            low=B_bound[0] * mu_R_abs_lower[i],
            high=B_bound[1] * mu_R_abs_lower[i],
            size=1,
        )
        mu_R_all[i] = mu_R[0]
        mu = np.array([mu_L, mu_R[0]], dtype=np.float32)
        if ar_model == "Gaussian":
            data[i, :] = GenDataMean(1, n, cp=tau_all[i], mu=mu, sigma=sigma)
        elif ar_model == "AR0":
            data[i, :] = GenDataMeanAR(1, n, cp=tau_all[i], mu=mu, sigma=1, coef=ARcoef)
        elif ar_model == "ARH":
            data[i, :] = GenDataMeanARH(
                1, n, cp=tau_all[i], mu=mu, coef=ARcoef, scale=scale
            )
        elif ar_model == "ARrho":
            data[i, :] = GenDataMeanARrho(1, n, cp=tau_all[i], mu=mu, sigma=sigma)
        elif ar_model == "Variance":
            if isinstance(sigma, list) and len(sigma) == 2:
                sigma_L, sigma_R = sigma
            else:
                raise ValueError("For 'Variance' type, sigma must be a list of two values: [sigma_L, sigma_R]")
            data[i, :] = GenDataVariance(1, n, cp=tau_all[i], mu=mu_L, sigma=[sigma_L, sigma_R])
    return {"data": data, "tau_alt": tau_all, "mu_R_alt": mu_R_all}

def MaxCUSUM(x):
    """
    To return the maximum of CUSUM

    Parameters
    ----------
    x : vector
        the time series

    Returns
    -------
    scalar
        the maximum of CUSUM
    """
    y = np.abs(ComputeCUSUM(x))
    return np.max(y)

def ComputeCUSUM(x):
    """
    Compute the CUSUM statistics with O(n) time complexity

    Parameters
    ----------
    x : vector
        the time series

    Returns
    -------
    vector
        a: the CUSUM statistics vector.
    """
    n = len(x)
    mean_left = x[0]
    mean_right = np.mean(x[1:])
    a = np.repeat(0.0, n - 1)
    a[0,] = np.sqrt((n - 1) / n) * (mean_left - mean_right)
    for i in range(1, n - 1):
        mean_left = mean_left + (x[i] - mean_left) / (i + 1)
        mean_right = mean_right + (mean_right - x[i]) / (n - i - 1)
        a[i,] = np.sqrt((n - i - 1) * (i + 1) / n) * (mean_left - mean_right)

    return a

def detect_change_in_stream(stream, model, window_length,k ):

    num_windows = len(stream) - window_length + 1
    predictions = np.zeros(num_windows)
    consecutive_changes = 0
    detected_change = 0
    detected_change_windows = []
    for i in range(num_windows):
        window_data = stream[i: i + window_length]
        window_input = np.expand_dims(window_data, axis=0)
        logits = model.predict(window_input, verbose=0)
        output = np.argmax(logits, axis=1)
    
        predictions[i] = output  # Binary label (1=change, 0=no change)
        if predictions[i] == 1:
            #print(f"Change detected at window {i}")
            consecutive_changes += 1
        else: 
            consecutive_changes = 0
        if consecutive_changes >= k:
            detected_change = 1
            detected_change_windows.append(i)
            #print(f"Change detected at window {i}")
        
    return detected_change#, detected_change_windows
#print(DataGenAlternative(1,,mu_L,n=100,B_bound=B_bound,sigma=1,ar_model="Gaussian"))