import numpy as np
import matplotlib.pyplot as plt


def x_true(t):
    return np.cos(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)


def sample(f, fs, duration):
    """Return (t_n, x_n) with t_n = n/fs, n = 0..N-1, N = fs*duration."""
    N = int(round(fs * duration))
    n = np.arange(N)
    t_n = n / fs
    return t_n, f(t_n)


def sinc_reconstruct(x_n, T, t):
    """xr(t) = sum_n x[n] * sinc((t - nT)/T), evaluated at every point of t.
    np.sinc is the normalized sinc: sin(pi x)/(pi x), same as the lecture."""
    n = np.arange(len(x_n))
    return np.sinc((t[:, None] - n[None, :] * T) / T) @ x_n


def metrics(xr, x, t, lo=1.0, hi=3.0):
    """MSE and max abs error over the interior lo <= t <= hi only."""
    mask = (t >= lo) & (t <= hi)
    err = xr[mask] - x[mask]
    return np.mean(err ** 2), np.max(np.abs(err))


if __name__ == "__main__":
    duration = 4
    t = np.arange(0, duration, 0.005)
    x_ref = x_true(t)

    f_max = 5                                   # highest tone in x(t), Hz
    print("Nyquist rate:", 2 * f_max, "Hz")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, fs in zip(axes, (20, 8)):
        t_n, x_n = sample(x_true, fs, duration)
        xr = sinc_reconstruct(x_n, 1 / fs, t)
        mse, mx = metrics(xr, x_ref, t)
        print(f"fs={fs}: MSE={mse:.6f}  Max error={mx:.6f}")

        ax.plot(t, x_ref, label="true x(t)")
        ax.plot(t, xr, "--", label="reconstruction")
        ax.plot(t_n, x_n, "ko", markersize=4, label="samples")
        ax.set_title(f"fs = {fs} Hz")
        ax.set_xlabel("t (s)")
        ax.legend()
    plt.tight_layout()
    plt.show()