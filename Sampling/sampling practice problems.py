"""
Sampling Theory — Practice Problems (with solutions)
======================================================
6 problems, roughly increasing in difficulty, matching typical
online-assessment styles for this topic. Each problem has:
  - PROBLEM statement (read this first, try it yourself)
  - SOLUTION code (check yourself after attempting)

Run this file directly to see every solution's output.
"""

import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# PROBLEM 1 — Find the Nyquist rate and check a given fs
# =================================================================
"""
A signal is x(t) = 3cos(2*pi*20*t) + 5sin(2*pi*45*t).
(a) Find the Nyquist rate.
(b) Is fs = 80 Hz sufficient to avoid aliasing? What about fs = 60 Hz?
"""

def problem1_solution():
    freqs = [20, 45]
    nyquist = 2 * max(freqs)
    print(f"[P1] Nyquist rate = {nyquist} Hz")
    for fs in [80, 60]:
        aliased = fs <= nyquist
        print(f"[P1] fs={fs} Hz -> {'ALIASING' if aliased else 'safe, no aliasing'}")


# =================================================================
# PROBLEM 2 — Identify the ALIASED frequency after undersampling
# =================================================================
"""
A pure tone at f0 = 30 Hz is sampled at fs = 22 Hz (below Nyquist,
so it WILL alias). What frequency does it appear as after sampling?

Formula: the apparent (aliased) frequency is the original frequency
"folded" into the range [0, fs/2] using:
    f_alias = | f0 - k*fs |   choosing integer k that lands f_alias
              in [0, fs/2]
"""

def find_aliased_frequency(f0, fs):
    """Fold f0 into the range [0, fs/2] by subtracting multiples of fs."""
    f_mod = f0 % fs                  # bring into [0, fs)
    if f_mod > fs / 2:
        f_mod = fs - f_mod           # fold down if in upper half
    return f_mod


def problem2_solution():
    f0, fs = 30, 22
    f_alias = find_aliased_frequency(f0, fs)
    print(f"[P2] True freq={f0} Hz sampled at fs={fs} Hz "
          f"-> appears as {f_alias} Hz")

    # Visual proof: plot true tone vs. what the samples "look like"
    t_fine = np.linspace(0, 1, 5000)
    x_true = np.cos(2 * np.pi * f0 * t_fine)
    n = np.arange(0, 1, 1 / fs)
    x_samples = np.cos(2 * np.pi * f0 * n)
    x_alias_curve = np.cos(2 * np.pi * f_alias * t_fine)  # what it "looks like"

    plt.figure(figsize=(7, 3))
    plt.plot(t_fine, x_true, alpha=0.3, label=f"true {f0} Hz")
    plt.plot(t_fine, x_alias_curve, 'g--', label=f"apparent {f_alias} Hz")
    plt.stem(n, x_samples, linefmt='r-', markerfmt='ro', basefmt=' ',
             label=f"samples @ fs={fs} Hz")
    plt.legend()
    plt.title("Problem 2: aliased frequency")
    plt.tight_layout()
    plt.show()


# =================================================================
# PROBLEM 3 — Compute the DFT manually and verify against np.fft.fft
# =================================================================
"""
Given x[n] = [1, 2, 3, 4] (N=4), compute the DFT X[k] using the
formula X[k] = sum_n x[n] * exp(-j*2*pi*k*n/N) directly (no library
FFT), then verify it matches numpy's fft.
"""

def dft_manual(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
    return X


def problem3_solution():
    x = np.array([1, 2, 3, 4])
    X_manual = dft_manual(x)
    X_numpy = np.fft.fft(x)
    print("[P3] Manual DFT: ", np.round(X_manual, 4))
    print("[P3] np.fft.fft: ", np.round(X_numpy, 4))
    print("[P3] Match:", np.allclose(X_manual, X_numpy))


# =================================================================
# PROBLEM 4 — Verify the IDFT reconstructs x[n] exactly
# =================================================================
"""
Using X[k] from Problem 3, compute the inverse DFT
    x[n] = (1/N) * sum_k X[k] * exp(j*2*pi*k*n/N)
and confirm you get back the original x[n] = [1,2,3,4].
"""

def idft_manual(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)
        x[n] /= N
    return x


def problem4_solution():
    x_original = np.array([1, 2, 3, 4])
    X = np.fft.fft(x_original)
    x_reconstructed = idft_manual(X)
    print("[P4] Original:      ", x_original)
    print("[P4] Reconstructed: ", np.round(x_reconstructed.real, 6))


# =================================================================
# PROBLEM 5 — Show spectral copies overlapping (aliasing in frequency domain)
# =================================================================
"""
A signal is band-limited with highest frequency fM = 40 Hz.
Plot the sampled spectrum's copies for:
   (a) fs = 100 Hz (oversampled -> copies separated)
   (b) fs = 60 Hz  (undersampled -> copies overlap)
Show both cases on the same style of plot as slide 17-18, but in
the frequency domain (copies of a triangular "spectrum shape").
"""

def triangular_spectrum(f, fM, height=1.0):
    """A simple triangular stand-in for X(jw): peak at 0, zero beyond fM."""
    return np.clip(height * (1 - np.abs(f) / fM), 0, None)


def plot_spectral_copies(fM, fs, f_range=150):
    f = np.linspace(-f_range, f_range, 2000)
    total = np.zeros_like(f)
    k_max = int(np.ceil(f_range / fs)) + 1
    plt.figure(figsize=(7, 3))
    for k in range(-k_max, k_max + 1):
        copy = triangular_spectrum(f - k * fs, fM) / (1)  # scaled by 1/T conceptually
        total += copy
        plt.plot(f, copy, 'b--', alpha=0.4)
    plt.plot(f, total, 'r', linewidth=1.5, label="sum (Xp(jw))")
    overlap = fs < 2 * fM
    plt.title(f"fs={fs} Hz, fM={fM} Hz -> "
              f"{'OVERLAP (aliasing)' if overlap else 'separated (safe)'}")
    plt.xlabel("Frequency (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def problem5_solution():
    fM = 40
    plot_spectral_copies(fM, fs=100)  # safe: fs > 2*fM = 80
    plot_spectral_copies(fM, fs=60)   # aliasing: fs < 80


# =================================================================
# PROBLEM 6 — Reconstruction error vs. sampling rate
# =================================================================
"""
For x(t) = sin(2*pi*10*t), compute the reconstruction error
(mean squared error between true signal and sinc-reconstructed
signal) for several sampling rates below and above the Nyquist
rate (20 Hz). Plot error vs. fs to show the sharp drop once
fs crosses the Nyquist rate.
"""

def sinc_reconstruct(n_samples, x_samples, T, t_eval):
    t_eval = np.asarray(t_eval)
    xr = np.zeros_like(t_eval, dtype=float)
    for nT, xn in zip(n_samples, x_samples):
        xr += xn * np.sinc((t_eval - nT) / T)
    return xr


def problem6_solution():
    f0 = 10
    nyquist = 2 * f0
    fs_list = [8, 12, 16, 18, 20, 25, 30, 40, 60]
    t_eval = np.linspace(0, 1, 3000)
    x_true = np.sin(2 * np.pi * f0 * t_eval)

    errors = []
    for fs in fs_list:
        n = np.arange(0, 1, 1 / fs)
        xs = np.sin(2 * np.pi * f0 * n)
        xr = sinc_reconstruct(n, xs, 1 / fs, t_eval)
        mse = np.mean((x_true - xr) ** 2)
        errors.append(mse)
        print(f"[P6] fs={fs:>3} Hz | MSE={mse:.5f}")

    plt.figure(figsize=(7, 3))
    plt.plot(fs_list, errors, 'o-')
    plt.axvline(nyquist, color='r', linestyle='--', label=f"Nyquist={nyquist} Hz")
    plt.yscale('log')
    plt.xlabel("Sampling rate fs (Hz)")
    plt.ylabel("MSE (log scale)")
    plt.title("Reconstruction error vs. sampling rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =================================================================
if __name__ == "__main__":
    problem1_solution()
    problem2_solution()
    problem3_solution()
    problem4_solution()
    problem5_solution()
    problem6_solution()
