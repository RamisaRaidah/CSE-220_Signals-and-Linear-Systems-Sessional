"""
CSE220 Online 2 (Step Response)

Instructions:
- Copy (or import) your completed Signal and LTI_System classes from Offline 1.
- Implement the TODO functions below.
- Do NOT use numpy.convolve / scipy.signal / any built-in convolution.
"""

import numpy as np
import matplotlib.pyplot as plt


# Paste/Import your Offline 1 implementations here
# from your_offline1_file import Signal, LTI_System

class Signal:
    def __init__(self, INF):
        # TODO: paste your Offline 1 implementation
        # raise NotImplementedError
        self.INF=INF
        self.values=np.full(2*INF+1,0.0)

    def set_value_at_time(self, t, value):
        if(t>=-self.INF and t<=self.INF):
            self.values[t+self.INF]=value
        else: 
            raise RuntimeError("t is out of range of this discrete signal")

        
    def get_value_at_time(self, t):
        if(t>=-self.INF and t<=self.INF):
            return self.values[t+self.INF]*1.0
        else: 
            return 0.0
        
    def shift(self, k):
        y = Signal(self.INF)        
        t = -self.INF + k              # shifted position of the first sample
        for val in self.values:
            if -self.INF <= t <= self.INF:   # drop anything shifted outside the window
                y.set_value_at_time(t, val)
            t += 1
        return y

    def add(self, other):
        start=max(self.INF,other.INF)
        y=Signal(start)

        t=-start
        for i in range (2*start+1):
            y.set_value_at_time(t,self.get_value_at_time(t)+other.get_value_at_time(t))
            t+=1
        return y
    
    def multiply(self, scalar):
        y=Signal(self.INF)
        for t in range(-(self.INF),self.INF+1):
            y.set_value_at_time(t,self.get_value_at_time(t)*scalar)
        return y

    def plot(self, title="Discrete Signal"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(-self.INF,self.INF+1),self.values)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title(title)
        ax.grid(True)
        return ax            
                

class LTI_System:
    def __init__(self, impulse_response: Signal):
        # TODO: paste your Offline 1 implementation
        self.impulse_response=impulse_response

    def linear_combination_of_impulses(self, input_signal: Signal):
        coeffs = []
        shifts = []
        for t in range(-input_signal.INF, input_signal.INF + 1):
            coeffs.append(input_signal.get_value_at_time(t))
            shifts.append(t)
        return coeffs, shifts
 
    def output(self, input_signal: Signal):
        y = Signal(input_signal.INF)
        for i in range(-input_signal.INF, input_signal.INF + 1):
            total = 0.0
            for j in range(-self.impulse_response.INF, self.impulse_response.INF + 1):
                total += input_signal.get_value_at_time(j) * self.impulse_response.get_value_at_time(i - j)
            y.set_value_at_time(i, total)
        return y
    
def read_signal_from_file(filename: str, INF: int) -> Signal:
    sig = Signal(INF)
    with open(filename, "r", encoding="utf-8") as f:
        nstart, nend = map(int, f.readline().strip().split())
        vals = list(map(float, f.readline().strip().split()))
    assert len(vals) == (nend - nstart + 1)
    for i, v in enumerate(vals):
        sig.set_value_at_time(nstart + i, v)
    return sig


def first_difference(sig: Signal) -> Signal:
    """
    Returns Δsig[n] = sig[n] - sig[n-1] (assume outside range is 0).
    Must use Signal.shift/add/multiply.
    """
    # TODO
    y=sig.multiply(-1)
    y=y.shift(1)
    y=y.add(sig)

    return y

def impulse_from_step_response(step_response: Signal) -> Signal:
    """
    Given s[n], compute h[n] = s[n] - s[n-1] (with s[-1]=0).
    Must use only Signal operations.
    """
    # TODO
    y=step_response.multiply(-1)
    y=y.shift(1)
    y=y.add(step_response)
    return y


def output_using_step_response(x: Signal, step_response: Signal) -> Signal:
    """
    Compute y[n] using ONLY step response:
        y = (Δx * s)
    You must reuse your Offline 1 LTI_System machinery (linear combination of impulses).
    """
    dx = first_difference(x)
    lti = LTI_System(step_response)
    return lti.output(dx)

# Main (demo workflow)
if __name__ == "__main__":
    # Choose INF large enough for your signals
    INF = 50

    # ---- Load provided files ----
    s = read_signal_from_file("step_response.txt", INF)
    x = read_signal_from_file("input_signal.txt", INF)

    # ---- Part 1: recover impulse response ----
    h = impulse_from_step_response(s)

    s.plot("Step Response s[n]")
    h.plot("Recovered Impulse Response h[n] = s[n] - s[n-1]")

    # ---- Part 2: output using only step response ----
    dx = first_difference(x)
    y_s = output_using_step_response(x,s)

    x.plot("Input x[n]")
    dx.plot("First Difference Δx[n]")
    y_s.plot("Output y_s[n] computed via step response")

    # ---- Part 3: verify with impulse-response method ----
    sys_h = LTI_System(h)
    y_h = sys_h.output(x)
    y_h.plot("Output y_h[n] computed via impulse response")

    # Check if outputs match closely
    if np.allclose(y_s.values, y_h.values, atol=1e-6):
        print("Outputs match closely!")
    else:
        print("Outputs differ!")
