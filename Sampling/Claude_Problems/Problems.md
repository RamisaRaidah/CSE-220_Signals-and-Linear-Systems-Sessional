## Mock 1: Rebuild the Signal (about 25 min, plots)

**Scenario:** A sensor records `x(t) = cos(2π·2t) + 0.5·sin(2π·5t)` for 4 seconds. You will sample it, rebuild it with ideal interpolation, and see what happens at a good and a bad sampling rate.

**Task 1: Sampling.** Write `sample(f, fs, duration)` that returns `(t_n, x_n)` with `t_n = n/fs` for `n = 0 … N−1` and   `N = fs·duration`.

**Task 2: Reconstruction.** Write `sinc_reconstruct(x_n, T, t)` using

`xr(t) = Σ x[n] · sinc((t − nT)/T)`, where `sinc(x) = sin(πx)/(πx)`.

It should return `xr` evaluated at every point of the array `t`.

**Task 3: Metrics.** Use the evaluation grid `t = np.arange(0, 4, 0.005)`. Over the interior `1 ≤ t ≤ 3` only, compute:

* MSE = mean of `(xr(t) − x(t))²`
* Max error = `max |xr(t) − x(t)|`

The edges are excluded because the finite sum is truncated there.

**Task 4: Plot and print.** Do this for `fs = 20` and `fs = 8`. Draw two subplots side by side, one per `fs`. Each subplot should show the true signal, the sample dots and the reconstruction. Print the Nyquist rate of the signal and both metrics for each `fs`.

**Self-check (approximate):**

```
fs=20  MSE ≈ 1e-5    Max ≈ 0.0054
fs=8   MSE ≈ 0.2497  Max ≈ 0.9758
```

---

## Mock 2: Alias Detective (about 20 min, pure functions)

**Scenario:** A logger samples cosine tones at `fs = 10 Hz`. You must predict what frequency each tone appears as after sampling.

**Task 1.** Write `apparent_frequency(f, fs) = |f − round(f/fs)·fs|`.

**Task 2.** Write `verify(f, fs)`. It builds `n = 0…49` and returns `True` if `cos(2π·f·n/fs)` equals `cos(2π·apparent_frequency(f, fs)·n/fs)` (use `np.allclose`).

**Task 3.** Write `nyquist_rate(freqs)`, which returns the minimum sampling rate needed for a signal containing those tones.

**Task 4.** For tones `[2, 7, 12, 18]` at `fs = 10`, print one line per tone: the tone, its apparent frequency, `OK` or `ALIASED`, and the verify result. Then print `nyquist_rate([3, 11, 25])`.

**Self-check:**

```
apparent frequencies: [2, 3, 2, 2]
ALIASED tones:        7, 12, 18
verify:               True for all four
nyquist_rate([3, 11, 25]) = 50
```

---

## Mock 3: The Resampler (about 30 min, template file)

**Scenario:** A recording was sampled with interval `T_in`. You need the same signal at a new interval `T_out`, using ideal (sinc) interpolation. Fill in the template below and submit a single file named with your student ID.

**Task 1.** `resample(x, T_in, T_out)` returns `(t_q, y)`.

* The output times are `t_q[m] = m·T_out` for every `m` with `t_q[m] ≤ (N−1)·T_in`.
* Each output value is `y[m] = Σ x[n] · sinc((t_q[m] − n·T_in)/T_in)`.

**Task 2.** `check_interpolation(x, T_in)` resamples with `T_out = T_in/2`. It returns the maximum absolute difference between the original samples and the resampled values at the original sample times, which are the even indices of `y`.

python

```python
import numpy as np

def resample(x, T_in, T_out):
    # implement
    pass

def check_interpolation(x, T_in):
    # implement
    pass

if __name__ == "__main__":
    x = [0, 1, 0, -1, 0, 1, 0, -1]
    t_q, y = resample(x, 1.0, 0.5)
    print("Result:", np.round(y, 4))
    print("Check:", check_interpolation(x, 1.0))
```

**Expected output:**

```
Result: [ 0.      0.5311  1.      0.8158 -0.     -0.7922 -1.     -0.6306  0.
  0.6306  1.      0.7922  0.     -0.8158 -1.    ]
Check: ~0 (below 1e-12)
```

That is 15 values, and the check is essentially zero. This confirms the fact from before: the sinc sum passes exactly through the original samples.
