"""
CFT Verification: Phase Shift (Modulation) + Time Compression (Scaling)
Signal: x(t) = Square(t) + Triangle(t)
No FFT used — pure numerical integration via np.trapz.
"""

import numpy as np
from scipy import signal as sp_signal
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. Signal generation
# ---------------------------------------------------------------
class SignalGenerator:
    """Generates x(t) = Square(t) + Triangle(t) for arbitrary time arrays."""

    def __init__(self, freq: float = 1.0):
        self.freq = freq  # fundamental frequency of both waveforms

    def square(self, t):
        return sp_signal.square(2 * np.pi * self.freq * t)

    def triangle(self, t):
        # width=0.5 -> symmetric triangle wave
        return sp_signal.sawtooth(2 * np.pi * self.freq * t, width=0.5)

    def x(self, t):
        return self.square(t) + self.triangle(t)


# ---------------------------------------------------------------
# 2. Transformations: phase modulation + time compression
# ---------------------------------------------------------------
class SignalTransformer:
    """
    Applies (ii) time compression then (i) phase modulation, in that
    order, so that y(t) = x(a t) * exp(j*2*pi*f0*t)
    This ordering is required to obtain Y(f) = (1/|a|) X((f-f0)/a).
    """

    def __init__(self, generator: SignalGenerator, f0: float, a: float):
        self.generator = generator
        self.f0 = f0
        self.a = a

    def compress(self, t):
        """w(t) = x(a t)"""
        return self.generator.x(self.a * t)

    def modulate(self, w, t):
        """apply phase shift 2*pi*f0*t via complex exponential"""
        return w * np.exp(1j * 2 * np.pi * self.f0 * t)

    def y(self, t):
        w = self.compress(t)
        return self.modulate(w, t)


# ---------------------------------------------------------------
# 3. Continuous Fourier Transform via numerical integration
# ---------------------------------------------------------------
class ContinuousFourierTransform:
    """
    X(f) = ∫ x(t) exp(-j*2*pi*f*t) dt
    Computed with np.trapz — no FFT anywhere.
    """

    def __init__(self, t: np.ndarray):
        self.t = t

    def transform(self, x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        # Build (Nt, Nf) matrix of kernels, integrate over t (axis=0)
        T = self.t[:, None]                      # (Nt, 1)
        F = freqs[None, :]                        # (1, Nf)
        kernel = np.exp(-1j * 2 * np.pi * F * T)   # (Nt, Nf)
        integrand = x[:, None] * kernel            # (Nt, Nf)
        X = np.trapz(integrand, self.t, axis=0)     # (Nf,)
        return X


# ---------------------------------------------------------------
# 4. Error analysis
# ---------------------------------------------------------------
class ErrorAnalyzer:
    @staticmethod
    def mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------
class Visualizer:
    @staticmethod
    def plot_magnitude(freqs, mag_Y, mag_Xshift, a):
        plt.figure(figsize=(9, 5))
        plt.plot(freqs, mag_Y, label=r'$|Y(f)|$', linewidth=2)
        plt.plot(freqs, mag_Xshift, '--',
                  label=r'$\frac{1}{|a|}|X((f-f_0)/a)|$', linewidth=2)
        plt.title('Magnitude Spectrum Verification')
        plt.xlabel('Frequency f'); plt.ylabel('Magnitude')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/magnitude_verification.png', dpi=150)
        plt.close()

    @staticmethod
    def plot_phase(freqs, phase_Y, phase_Xshift):
        plt.figure(figsize=(9, 5))
        plt.plot(freqs, phase_Y, label=r'$\angle Y(f)$', linewidth=2)
        plt.plot(freqs, phase_Xshift, '--',
                  label=r'$\angle X((f-f_0)/a)$', linewidth=2)
        plt.title('Phase Spectrum Verification')
        plt.xlabel('Frequency f'); plt.ylabel('Phase (rad)')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/phase_verification.png', dpi=150)
        plt.close()

    @staticmethod
    def plot_time_domain(t, x_t, y_t):
        fig, axs = plt.subplots(2, 1, figsize=(9, 6))
        axs[0].plot(t, x_t)
        axs[0].set_title('x(t) = Square(t) + Triangle(t)')
        axs[0].grid(True)
        axs[1].plot(t, np.real(y_t), label='Re[y(t)]')
        axs[1].plot(t, np.imag(y_t), label='Im[y(t)]')
        axs[1].set_title('y(t) = x(a t) * exp(j2πf0t)')
        axs[1].legend(); axs[1].grid(True)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/time_domain_signals.png', dpi=150)
        plt.close()


# ---------------------------------------------------------------
# 6. Main experiment
# ---------------------------------------------------------------
class Experiment:
    def __init__(self, f0=10.0, a=10.0,
                 t_start=-5, t_end=5, n_t=4001,
                 f_start=-10, f_end=10, n_f=1001):
        self.f0 = f0
        self.a = a
        self.t = np.linspace(t_start, t_end, n_t)
        self.freqs = np.linspace(f_start, f_end, n_f)

        self.generator = SignalGenerator(freq=1.0)
        self.transformer = SignalTransformer(self.generator, f0, a)
        self.cft = ContinuousFourierTransform(self.t)

    def run(self):
        # Time-domain signals
        x_t = self.generator.x(self.t)
        y_t = self.transformer.y(self.t)

        # X(f) evaluated directly at shifted/scaled frequency (f-f0)/a
        freqs_shift = (self.freqs - self.f0) / self.a
        X_shift = self.cft.transform(x_t, freqs_shift)
        rhs = (1.0 / abs(self.a)) * X_shift          # (1/|a|) X((f-f0)/a)

        # Y(f) evaluated directly from y(t)
        Y = self.cft.transform(y_t, self.freqs)

        mag_Y, mag_rhs = np.abs(Y), np.abs(rhs)
        phase_Y, phase_rhs = np.angle(Y), np.angle(rhs)

        mse_mag = ErrorAnalyzer.mse(mag_Y, mag_rhs)
        mse_phase = ErrorAnalyzer.mse(phase_Y, phase_rhs)

        Visualizer.plot_time_domain(self.t, x_t, y_t)
        Visualizer.plot_magnitude(self.freqs, mag_Y, mag_rhs, self.a)
        Visualizer.plot_phase(self.freqs, phase_Y, phase_rhs)

        print(f"MSE (magnitude) = {mse_mag:.6e}")
        print(f"MSE (phase)     = {mse_phase:.6e}")

        return mse_mag, mse_phase


if __name__ == "__main__":
    exp = Experiment()
    exp.run()