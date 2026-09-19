"""
Road 1 vs Road 2 — two ways to compute Xp(jw)
================================================
Road 1: Xp(jw) = sum_n x(nT) * exp(-j*w*n*T)          [direct definition]
Road 2: Xp(jw) = (1/T) * sum_k X(j(w - k*ws))          [copies of X(jw)]

We use a Gaussian pulse because it has a clean, KNOWN analytic
Fourier transform (also a Gaussian) — that's what lets us actually
build Road 2's X(jw) term at all. This is the point: Road 2 needs
you to already know X(jw) analytically; Road 1 doesn't.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# -----------------------------------------------------------
# Signal: Gaussian pulse x(t) = exp(-t^2 / (2*sigma^2))
# Known CTFT: X(jw) = sigma * sqrt(2*pi) * exp(-sigma^2 * w^2 / 2)
# -----------------------------------------------------------
sigma = 0.05

def x_t(t):
    return np.exp(-t**2 / (2 * sigma**2))

def X_jw_analytic(w):
    return sigma * np.sqrt(2 * np.pi) * np.exp(-sigma**2 * w**2 / 2)


# -----------------------------------------------------------
# Sampling setup
# -----------------------------------------------------------
fs = 200                     # sampling rate (Hz)
T = 1 / fs
ws = 2 * np.pi * fs          # sampling freq in rad/s
n_range = np.arange(-100, 101)      # sample indices n
t_samples = n_range * T
x_samples = x_t(t_samples)

w_axis = np.linspace(-2 * ws, 2 * ws, 2000)   # frequency axis to evaluate on


# -----------------------------------------------------------
# ROAD 1: direct definition, Xp(jw) = sum_n x(nT) e^{-jwnT}
# -----------------------------------------------------------
def road1_direct_sum(w_axis, x_samples, t_samples):
    Xp = np.zeros_like(w_axis, dtype=complex)
    for i, w in enumerate(w_axis):
        Xp[i] = np.sum(x_samples * np.exp(-1j * w * t_samples))
    return Xp


# -----------------------------------------------------------
# ROAD 2: copies of X(jw), spaced ws apart, scaled by 1/T
# -----------------------------------------------------------
def road2_shifted_copies(w_axis, k_max=5):
    Xp = np.zeros_like(w_axis, dtype=complex)
    for k in range(-k_max, k_max + 1):
        Xp += X_jw_analytic(w_axis - k * ws)
    return Xp / T


# -----------------------------------------------------------
# Run both, time both, compare
# -----------------------------------------------------------
t0 = time.time()
Xp_road1 = road1_direct_sum(w_axis, x_samples, t_samples)
time_road1 = time.time() - t0

t0 = time.time()
Xp_road2 = road2_shifted_copies(w_axis, k_max=5)
time_road2 = time.time() - t0

print(f"Road 1 time: {time_road1:.5f} s")
print(f"Road 2 time: {time_road2:.5f} s")
print(f"Max difference between the two results: "
      f"{np.max(np.abs(np.abs(Xp_road1) - np.abs(Xp_road2))):.4f}")
print("(Road 2 needs a large enough k_max — try increasing it if this isn't small)")

# -----------------------------------------------------------
# Plot: they should overlay almost perfectly
# -----------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(w_axis, np.abs(Xp_road1), label="Road 1 (direct sum)", linewidth=2)
plt.plot(w_axis, np.abs(Xp_road2), '--', label="Road 2 (shifted copies)", linewidth=2)
plt.axvline(ws / 2, color='gray', linestyle=':', label="ws/2")
plt.axvline(-ws / 2, color='gray', linestyle=':')
plt.title("Xp(jw): Road 1 vs Road 2 (should match)")
plt.xlabel("w (rad/s)")
plt.ylabel("|Xp(jw)|")
plt.legend()
plt.tight_layout()
plt.show()
