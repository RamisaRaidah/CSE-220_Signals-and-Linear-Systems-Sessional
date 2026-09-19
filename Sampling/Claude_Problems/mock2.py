import numpy as np


def apparent_frequency(f, fs):
    """Frequency a tone f appears at after sampling at fs."""
    return abs(f - round(f / fs) * fs)


def verify(f, fs):
    """True if samples of cos at f equal samples of cos at its apparent frequency."""
    n = np.arange(0, 50)
    f_a = apparent_frequency(f, fs)
    return bool(np.allclose(np.cos(2 * np.pi * f * n / fs),
                            np.cos(2 * np.pi * f_a * n / fs)))


def nyquist_rate(freqs):
    """Minimum sampling rate for a signal containing these tones (Hz)."""
    return 2 * max(freqs)


if __name__ == "__main__":
    fs = 10
    tones = [2, 7, 12, 18]
    for f in tones:
        f_a = apparent_frequency(f, fs)
        status = "OK" if f < fs / 2 else "ALIASED"
        print(f"tone={f:>3} Hz  apparent={f_a:>3} Hz  {status:<8} verify={verify(f, fs)}")
    print("nyquist_rate([3, 11, 25]) =", nyquist_rate([3, 11, 25]))