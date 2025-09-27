#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

m_true = 0.25             # kg
fs = 200                  # Hz
dt = 1.0 / fs
T_total = 5.0             # seconds
N = int(T_total / dt)
g = 9.81                  # m/s^2

T_mean = 2.7              # N
sigma_T = np.sqrt(0.25)   # N (std from variance 0.25 N^2)

R_min, R_max = 0.01, 0.5  # m^2

def get_ABd(m, dt):
    """ Discrete-time exact ZOH for constant acceleration with gravity. """
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[dt**2/(2.0*m)],
                  [dt/m]])
    d = np.array([[-0.5*g*dt**2],
                  [-g*dt]])
    return A, B, d

def simulate_truth_and_measurements(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N+1) * dt
    x_true = np.zeros((2, N+1))  # [h; v]
    z_meas = np.zeros(N+1)
    R_seq = np.zeros(N+1)

    A_true, B_true, d_true = get_ABd(m_true, dt)
    for k in range(N):
        # thrust with noise for the true system
        T_k = T_mean + rng.normal(0.0, sigma_T)
        # propagate true state
        x_true[:, k+1] = (A_true @ x_true[:, k] + (B_true[:,0] * T_k) + d_true[:,0])
        # measurement noise variance (reported correctly)
        R_seq[k+1] = rng.uniform(R_min, R_max)
        z_meas[k+1] = x_true[0, k+1] + rng.normal(0.0, np.sqrt(R_seq[k+1]))
    return t, x_true, z_meas, R_seq

def run_kf(m_est, z, R_seq, dt, T_mean, sigma_T, x0=None, P0=None):
    """ Standard discrete-time KF for the 2-state model. """
    A, B, d = get_ABd(m_est, dt)
    Sigma_u = np.array([[sigma_T**2]])     
    Q = B @ Sigma_u @ B.T                   

    if x0 is None:
        x_hat = np.zeros(2)
    else:
        x_hat = np.array(x0, dtype=float).copy()

    if P0 is None:
        P = np.diag([10.0, 10.0])           
    else:
        P = np.array(P0, dtype=float).copy()

    H = np.array([[1.0, 0.0]])

    x_hist = np.zeros((2, N+1))
    x_hist[:, 0] = x_hat
    for k in range(N):
        # Predict
        u_k = T_mean
        x_hat = (A @ x_hat) + (B[:,0] * u_k) + d[:,0]
        P = A @ P @ A.T + Q

        # Update with varying R_k
        Rk = np.array([[R_seq[k+1]]])
        y = z[k+1] - (H @ x_hat)             
        S = H @ P @ H.T + Rk
        K = P @ H.T @ np.linalg.inv(S)
        x_hat = x_hat + (K[:,0] * y)       
        P = (np.eye(2) - K @ H) @ P
        x_hist[:, k+1] = x_hat
    return x_hist

def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))

def main():
    t, x_true, z_meas, R_seq = simulate_truth_and_measurements(seed=42)

    # truth
    alt_truth = x_true[0]

    alt_raw = z_meas.copy()

    # KF matched
    x_kf_matched = run_kf(m_true, z_meas, R_seq, dt, T_mean, sigma_T)
    alt_kf_matched = x_kf_matched[0]

    # KF with mass +10%
    m_est = 1.10 * m_true
    x_kf_mismatch = run_kf(m_est, z_meas, R_seq, dt, T_mean, sigma_T)
    alt_kf_mismatch = x_kf_mismatch[0]

    # Errors
    err_raw = alt_raw - alt_truth
    err_kf_matched = alt_kf_matched - alt_truth
    err_kf_mismatch = alt_kf_mismatch - alt_truth

    # Print quick metrics
    print("RMSE (altitude, w.r.t. truth):")
    print(f"  Raw sensor     : {rmse(alt_raw, alt_truth):.4f} m")
    print(f"  KF (matched)   : {rmse(alt_kf_matched, alt_truth):.4f} m")
    print(f"  KF (mass +10%) : {rmse(alt_kf_mismatch, alt_truth):.4f} m")

    # Plot altitude traces
    plt.figure(figsize=(10, 5))
    plt.plot(t, alt_truth, label='(a) Ground truth')
    plt.plot(t, alt_raw, label='(b) Sensor (raw)')
    plt.plot(t, alt_kf_matched, label='(c) KF (matched)')
    plt.plot(t, alt_kf_mismatch, label='(d) KF (mass +10%)')
    plt.xlabel('Time [s]')
    plt.ylabel('Altitude [m]')
    plt.title('Drone liftoff: altitude (5 s at 200 Hz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('drone_altitude_traces.png', dpi=160)

    # Plot errors
    plt.figure(figsize=(10, 5))
    plt.plot(t, err_raw, label='Raw - Truth')
    plt.plot(t, err_kf_matched, label='KF matched - Truth')
    plt.plot(t, err_kf_mismatch, label='KF mass+10% - Truth')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.title('Estimation errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig('drone_altitude_errors.png', dpi=160)

    print("Saved figures: 'drone_altitude_traces.png', 'drone_altitude_errors.png'")

if __name__ == "__main__":
    main()
