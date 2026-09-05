import math
import cmath


def fft(a):
    """Recursive radix-2 Cooley-Tukey FFT. len(a) must be a power of 2."""
    n = len(a)
    if n == 1:
        return a
    even = fft(a[0::2])
    odd = fft(a[1::2])
    result = [0] * n
    for k in range(n // 2):
        t = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        result[k] = even[k] + t
        result[k + n // 2] = even[k] - t
    return result


def ifft(a):
    """Inverse FFT via conjugate trick."""
    n = len(a)
    conj = [x.conjugate() for x in a]
    y = fft(conj)
    return [x.conjugate() / n for x in y]


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def weighted_polynomial_multiply(P, Q, W):
    """
    P, Q, W are given in descending-power order (as in the problem statement),
    e.g. P(x) = 1x^2 + 3x + 2 -> P = [1, 3, 2].

    R[k] = sum_{i=0}^{k} w_i * p_i * q_{k-i},  k = 0 .. m+n
    (indices here are ascending powers: p_0 is the constant term, etc.)

    Trick: reverse P, Q, W to ascending order, fold w_i into p_i
    (p'_i = w_i * p_i), then R is just the ordinary convolution of p' and q,
    computed via FFT/IFFT circular convolution with zero-padding.
    """
    m = len(P) - 1
    n = len(Q) - 1

    # convert descending-power lists to ascending-power lists
    p = P[::-1]   # p[i] = coefficient of x^i
    q = Q[::-1]   # q[i] = coefficient of x^i
    w = W[::-1]   # w[i] = weight for index i

    # fold the weight into p: p'_i = w_i * p_i
    p_weighted = [w[i] * p[i] for i in range(len(p))]

    result_len = m + n + 1          # number of coefficients in R
    size = _next_pow2(result_len)   # FFT size (power of 2, big enough to avoid wraparound)

    a = p_weighted + [0] * (size - len(p_weighted))
    b = q + [0] * (size - len(q))

    fa = fft(a)
    fb = fft(b)
    fc = [fa[i] * fb[i] for i in range(size)]
    c = ifft(fc)

    # extract the relevant coefficients and round to nearest integer
    R = [round(c[i].real) for i in range(result_len)]
    return R


if __name__ == "__main__":
    P = [1, 3, 2, 6, 7]
    Q = [4, 1]
    W = [3, 2, 1, 5, 6]

    R = weighted_polynomial_multiply(P, Q, W)

    print("Result:", R)