import numpy as np


def resample(x, T_in, T_out):
    """Sinc-resample x (spacing T_in) onto a grid with spacing T_out.
    Output times: t_q[m] = m*T_out for all m with t_q[m] <= (N-1)*T_in."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    t_max = (N - 1) * T_in
    M = int(np.floor(t_max / T_out + 1e-9)) + 1
    t_q = np.arange(M) * T_out
    n = np.arange(N)
    y = np.sinc((t_q[:, None] - n[None, :] * T_in) / T_in) @ x
    return t_q, y


def check_interpolation(x, T_in):
    """Max |x[n] - y at original sample times|, using T_out = T_in/2."""
    x = np.asarray(x, dtype=float)
    _, y = resample(x, T_in, T_in / 2)
    return np.max(np.abs(y[::2] - x))


if __name__ == "__main__":
    x = [0, 1, 0, -1, 0, 1, 0, -1]
    t_q, y = resample(x, 1.0, 0.5)
    print("Result:", np.round(y, 4))
    print("Check:", check_interpolation(x, 1.0))