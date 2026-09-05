import numpy as np

from svg_utils import load_svg_path
from epicycle_animation import save_outputs


class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        """
        Step 1: Store the sampled signal and set up everything the other
        methods will need.

        Parameters
        ----------
        t : 1D numpy array, shape (M,)
            Uniformly spaced sample times covering ONE FULL PERIOD of the
            signal, as a *closed* interval: t[0] == 0 and t[-1] == T (the
            period). This is exactly what svg_utils.load_svg_path(...)
            returns.
        signal : 1D complex numpy array, shape (M,)
            signal[i] = f(t[i]) = x(t[i]) + 1j * y(t[i]). Periodic, so
            signal[-1] == signal[0].
        n_harmonics : int (call it N)
            The series will use every integer harmonic n with
            -N <= n <= N (i.e. 2N+1 terms in total -- do not forget the
            negative harmonics).

        You must set at least the following attributes, since the rest of
        this class (and the provided plotting/animation code) expects
        them to exist:
            self.t, self.signal, self.N
            self.T      -- the period (a float)
            self.omega  -- the fundamental angular frequency, 2*pi/T
            self.coeffs -- an (initially empty) dict that will map
                           n -> c_n once calculate_all_coefficients() has
                           been called
        """
        self.t=t
        self.signal=signal
        self.N=n_harmonics
        self.T=t[-1]


        self.omega=(2*np.pi)/self.T
        self.coeffs={}


    def calculate_cn(self, n):
        """
        Step 2: Compute a single complex Fourier coefficient c_n using
        numerical integration (np.trapezoid) over the stored samples
        self.t, self.signal.

            c_n = (1/T) * integral_0^T  f(t) * exp(-j*n*omega*t)  dt

        n may be zero, positive, or negative.
        """
        return (1/self.T)*np.trapezoid(self.signal*np.exp(-1*(n*self.omega*self.t*1j)), self.t)

    def calculate_all_coefficients(self):
        """
        Step 3: Populate self.coeffs with c_n for every harmonic
        n = -N, ..., -1, 0, 1, ..., N by repeatedly calling calculate_cn(n).
        """
        for n in range(-self.N, self.N+1):
            self.coeffs[n]=self.calculate_cn(n)

        return self.coeffs

    def approximate(self, t):
        """
        Step 4: Reconstruct (an approximation of) the signal at time(s) t
        from the coefficients already stored in self.coeffs:

            f_hat(t) = sum_{n=-N}^{N} c_n * exp(j*n*omega*t)

        t may be a single number or a numpy array of times -- your
        implementation must support both, since the provided
        plotting/animation code calls this both ways.
        """
        cn=self.coeffs

        if(np.isscalar(t)):
            x=0
            for i in range(-self.N,self.N+1):
                x+=(cn[i]*np.exp(i*self.omega*t*1j))
            return x
        else:
            idx=0
            y=np.zeros(len(t), dtype=complex)
            for k in t:
                x=0+0j
                for i in range(-self.N,self.N+1):
                    x+=(cn[i]*np.exp(i*self.omega*k*1j))
                y[idx]=x
                idx+=1
            return y


    def prune_harmonics_by_energy(self,r):
        energy=0
        self.calculate_all_coefficients()

        arr = [(0, 0)] * len(self.coeffs)

        idx=0
        for i in range(-self.N,self.N+1):
            energy+=(np.abs(self.coeffs[i])**2)
            arr[idx]=(np.abs(self.coeffs[i]),i)
            idx+=1


        arr = sorted(arr, key=lambda p: p[0])
        cnt=0
        enn=0

        l=len(arr)
        idx=-self.N
        idx2=l-1
        for i in range(l):
            enn+=((arr[idx2][0]))**2
            
            if((np.abs(enn)/np.abs(energy))>r):
                self.coeffs[arr[idx2][1]]=0
            else: cnt+=1

            idx+=1
            idx2-=1
        

        en2=0
        for i in range(-self.N,self.N+1):
            en2+=((np.abs(self.coeffs[i])**2))

        return cnt, (np.abs(en2)/np.abs(energy))

    def evaluate_reconstruction_error(self):
        f_hat=self.approximate(t)

        mse=0+0j
        for i in range(len(self.signal)):
            mse+=(self.signal[i]-f_hat[i])**2

        mse/=len(self.signal)
        return mse




if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Usage: python3 assignment.py <path_to_svg> [n_harmonics] [comparison_png_path] [gif_path]
    if len(sys.argv) < 2:
        print("Usage: python3 assignment.py <path_to_svg> [n_harmonics] [comparison_png_path] [gif_path]")
        print("Example: python3 assignment.py svgs/heart.svg 150 heart_comparison.png heart_epicycles.gif")
        sys.exit(1)

    svg_path = sys.argv[1]
    N_HARMONICS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    stem = Path(svg_path).stem
    comparison_path = sys.argv[3] if len(sys.argv) > 3 else f"{stem}_comparison.png"
    gif_path = sys.argv[4] if len(sys.argv) > 4 else f"{stem}_epicycles.gif"

    t, z = load_svg_path(svg_path, num_points=1000)
    fs = FourierEpicycles(t, z, n_harmonics=N_HARMONICS)
    fs.calculate_all_coefficients()

    cnt1,r1=fs.prune_harmonics_by_energy(0.96)
    cnt2,r2=fs.prune_harmonics_by_energy(0.98)
    cnt3,r3=fs.prune_harmonics_by_energy(0.99)
    cnt4,r4=fs.prune_harmonics_by_energy(1)
    cnt5,r5=fs.prune_harmonics_by_energy(0.4)

    mse=fs.evaluate_reconstruction_error()

    print(f"0.96 {cnt1} {r1} {mse}")
    print(f"0.98 {cnt2} {r2} {mse}")
    print(f"0.99 {cnt3} {r3} {mse}")
    print(f"1.00 {cnt4} {r4} {mse}")
    print(f"0.4 {cnt5} {r5} {mse}")


    save_outputs(fs, z, comparison_path, gif_path, num_frames=240)
