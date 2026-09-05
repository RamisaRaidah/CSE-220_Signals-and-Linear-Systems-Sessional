"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    cnt=0
    n-=1
    while(n>0):
        cnt+=1
        n//=2

    return (1<<cnt)

def log2n(n):
    cnt=0
    n-=1
    while(n>0):
        cnt+=1
        n//=2
    return cnt

def bitReversal(n,k):
    res=0
    for _ in range(k):
        res=(res<<1)|(n&1)
        n=n>>1
    return res

class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        N=len(x)
        w=(2*np.pi)/N
        X=np.full(N,0.0j)
        for k in range(N):
            X[k]=0
            for n in range(N):
                wn=np.exp(-(1j)*(w*k*n))
                X[k]+=(x[n]*wn)

        return X

        

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        N=len(spectrum)
        w=(2*np.pi)/N
        x=np.full(N,0.0j)
        for n in range(N):
            for k in range(N):
                wn=np.exp((1j)*(w*k*n))
                x[n]+=(spectrum[k]*wn)
            x[n]/=N

        return x 


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def butterfly(self,x,l,M,wm):
        W=1
        for k in range(0,M//2):
            g=x[l+k]
            h=W*x[l+k+M//2]
            x[l+k]=g+h
            x[l+k+M//2]=g-h
            W=W*wm
        return x

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        N=len(x)
        if((N&(N-1)!=0) | (N==0)):
            raise ValueError(f"Length {N} is not a power of two")
        cnt=log2n(N)
        xf=np.full(N,0.0j)
        for i in range(N):
            xf[i]=x[i]

        x=np.full(N,0.0j)
        for i in range(N):
            x[i]=xf[bitReversal(i,cnt)]

        for s in range(1, cnt+1):
            M=(1<<s)
            wm=np.exp((-1j)*(2*np.pi)/M) 
            for l in range(0,N,M):
                x=self.butterfly(x,l,M,wm)        
        return x


    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        N=len(spectrum)
        if(N&(N-1)!=0 | N==0):
            raise ValueError(f"Length {N} is not a power of two")

        cnt=log2n(N)
        xf=np.full(N,0.0j)
        for i in range(N):
            xf[i]=spectrum[i]

        x=np.full(N,0.0j)
        for i in range(N):
            x[i]=xf[bitReversal(i,cnt)]

        for s in range(1, cnt+1):
            M=(1<<s)
            wm=np.exp((1j)*(2*np.pi)/M) 
            for l in range(0,N,M):
                x=self.butterfly(x,l,M,wm)  

        x=x/N    
        return x

# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.transform")

    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.inverse")
