"""
Sampling Theory — Reusable Templates
=====================================
Covers: signal generation, sampling, spectrum (FFT), aliasing demo,
reconstruction (sinc / ZOH / linear interpolation), Nyquist check.
 
Each function is self-contained — copy just the ones you need into
your assessment notebook.
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
 
 
# ---------------------------------------------------------------
# 1. GENERATE A CONTINUOUS-TIME SIGNAL (finely sampled to look "continuous")
# ---------------------------------------------------------------
def make_signal(freqs, amps, t_max=1.0, fs_fine=100000):
    """
    Build x(t) = sum(amps[i] * sin(2*pi*freqs[i]*t)) sampled very
    finely, to act as a stand-in for a true continuous signal.
 
    freqs : list of frequencies in Hz, e.g. [5, 50]
    amps  : list of amplitudes, same length as freqs
    fs_fine : sampling rate used just for plotting the "continuous" curve
    """
    t = np.arange(0, t_max, 1 / fs_fine)
    x = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return t, x
 
 
# ---------------------------------------------------------------
# 2. SAMPLE THE SIGNAL AT RATE fs  (this IS xp(t) conceptually)
# ---------------------------------------------------------------
def sample_signal(freqs, amps, fs, t_max=1.0):
    """
    Directly generate the DISCRETE samples x[n] = x(nT), T = 1/fs.
    This is what you'd actually be given/asked to produce in a problem.
    """
    n = np.arange(0, t_max, 1 / fs)
    x = np.zeros_like(n)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2 * np.pi * f * n)
    return n, x  # n here is actual time values nT, x is the samples
 
 
# ---------------------------------------------------------------
# 3. SPECTRUM VIA FFT  (numerical version of finding X(jw) / X(e^jw))
# ---------------------------------------------------------------
def plot_spectrum(x, fs, title="Spectrum"):
    """
    Plot magnitude spectrum of a sampled signal x, sampled at rate fs.
    Shows the repeated/aliased copies described in the lecture (slide 14).
    """
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))  # in Hz
 
    plt.figure(figsize=(7, 3))
    plt.plot(freqs, np.abs(X) / N)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return freqs, X
 
 
# ---------------------------------------------------------------
# 4. NYQUIST CHECK
# ---------------------------------------------------------------
def nyquist_rate(freqs):
    """
    Given the list of frequency components in a signal, return the
    Nyquist rate (minimum fs needed to avoid aliasing): 2 * f_max.
    """
    f_max = max(freqs)
    return 2 * f_max
 
 
def check_aliasing(freqs, fs):
    """
    Returns True if aliasing WILL occur when sampling at fs.
    (i.e. fs <= Nyquist rate)
    """
    return fs <= nyquist_rate(freqs)
 
 
# ---------------------------------------------------------------
# 5. VISUAL DEMO: OVERSAMPLING vs UNDERSAMPLING (slides 17-18)
# ---------------------------------------------------------------
def compare_sampling_rates(freqs, amps, fs_list, t_max=1.0):
    """
    Plot the time-domain samples for several sampling rates side by
    side, so you can visually see aliasing (a high-freq signal looking
    like a low-freq one when undersampled).
    """
    t_fine, x_fine = make_signal(freqs, amps, t_max)
 
    fig, axes = plt.subplots(len(fs_list), 1, figsize=(7, 3 * len(fs_list)))
    if len(fs_list) == 1:
        axes = [axes]
 
    for ax, fs in zip(axes, fs_list):
        n, xs = sample_signal(freqs, amps, fs, t_max)
        ax.plot(t_fine, x_fine, alpha=0.4, label="true x(t)")
        ax.stem(n, xs, linefmt="r-", markerfmt="ro", basefmt=" ",
                label=f"samples @ fs={fs} Hz")
        aliased = check_aliasing(freqs, fs)
        ax.set_title(f"fs={fs} Hz  |  Nyquist={nyquist_rate(freqs)} Hz  "
                     f"|  Aliasing: {'YES' if aliased else 'no'}")
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
 
 
# ---------------------------------------------------------------
# 6. RECONSTRUCTION METHOD 1: IDEAL SINC INTERPOLATION (slide 21)
# ---------------------------------------------------------------
def sinc_reconstruct(n_samples, x_samples, T, t_eval):
    """
    Ideal reconstruction: xr(t) = sum_n x(nT) * sinc((t - nT) / T)
    n_samples : sample time instants (nT values)
    x_samples : the sample values x(nT)
    T         : sampling period
    t_eval    : fine time grid on which to reconstruct xr(t)
    """
    t_eval = np.asarray(t_eval)
    xr = np.zeros_like(t_eval, dtype=float)
    for nT, xn in zip(n_samples, x_samples):
        xr += xn * np.sinc((t_eval - nT) / T)  # np.sinc(x) = sin(pi x)/(pi x)
    return xr
 
 
# ---------------------------------------------------------------
# 7. RECONSTRUCTION METHOD 2: ZERO-ORDER HOLD (slide 22)
# ---------------------------------------------------------------
def zoh_reconstruct(n_samples, x_samples, t_eval):
    """
    Staircase reconstruction: hold each sample constant until the next.
    """
    f = interpolate.interp1d(n_samples, x_samples, kind="previous",
                              fill_value="extrapolate")
    return f(t_eval)
 
 
# ---------------------------------------------------------------
# 8. RECONSTRUCTION METHOD 3: LINEAR INTERPOLATION (slide 26)
# ---------------------------------------------------------------
def linear_reconstruct(n_samples, x_samples, t_eval):
    """
    Connect the dots with straight lines (first-order hold).
    """
    f = interpolate.interp1d(n_samples, x_samples, kind="linear",
                              fill_value="extrapolate")
    return f(t_eval)
 
 
# ---------------------------------------------------------------
# 9. FULL WORKED EXAMPLE (run this file directly to see it work)
# ---------------------------------------------------------------
if __name__ == "__main__":
    freqs = [5]        # a 5 Hz sine wave
    amps = [1.0]
    T_max = 1.0
 
    print("Nyquist rate:", nyquist_rate(freqs), "Hz")
 
    # --- Oversampling vs undersampling demo ---
    compare_sampling_rates(freqs, amps, fs_list=[8, 30], t_max=T_max)
 
    # --- Spectrum before/after sampling at a safe rate ---
    fs = 50  # > Nyquist(=10 Hz), safe
    n, xs = sample_signal(freqs, amps, fs, T_max)
    plot_spectrum(xs, fs, title=f"Spectrum sampled at fs={fs} Hz")
 
    # --- Reconstruction comparison ---
    t_fine = np.linspace(0, T_max, 2000)
    x_true = amps[0] * np.sin(2 * np.pi * freqs[0] * t_fine)
 
    xr_sinc = sinc_reconstruct(n, xs, 1 / fs, t_fine)
    xr_zoh = zoh_reconstruct(n, xs, t_fine)
    xr_lin = linear_reconstruct(n, xs, t_fine)
 
    plt.figure(figsize=(8, 4))
    plt.plot(t_fine, x_true, 'k--', label="true x(t)", linewidth=1)
    plt.plot(t_fine, xr_sinc, label="sinc (ideal)")
    plt.plot(t_fine, xr_zoh, label="ZOH")
    plt.plot(t_fine, xr_lin, label="linear")
    plt.stem(n, xs, linefmt="r-", markerfmt="ro", basefmt=" ", label="samples")
    plt.legend()
    plt.title("Reconstruction methods compared")
    plt.tight_layout()
    plt.show()