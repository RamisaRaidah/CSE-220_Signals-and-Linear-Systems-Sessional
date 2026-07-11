import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

INF = 8

def plot(
        signal, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    # set y range of 
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    # plt.show()

def init_signal():
    return np.zeros(2 * INF + 1)


def time_scale_signal(x : np.ndarray, k : int) -> np.ndarray:
    # implement this function
    y=np.zeros_like(x)
    cnt=-8
    for i in x:
        if((cnt*k+INF)>=0 and (cnt*k+INF)>=0<x.size):
            y[(cnt*k+INF)]=i
    
    return y

def time_scale_signal_interpolate(x : np.ndarray, k : int) -> np.ndarray:
    # implement this function
    y = np.zeros_like(x, dtype=float)
 
    n = np.arange(-INF, INF + 1)
    mask = (n % k == 0)
 
    # --- exact (non-intermediate) samples ---
    src_n = n // k
    src_idx = src_n + INF
    valid_direct = mask & (src_idx >= 0) & (src_idx < x.size)
    out_idx_direct = n[valid_direct] + INF
    y[out_idx_direct] = x[src_idx[valid_direct]]
 
    # --- intermediate samples: average of neighboring original samples ---
    n1 = n // k          # floor(n/k)
    n2 = n1 + 1
    idx1 = n1 + INF
    idx2 = n2 + INF
 
    idx1_clipped = np.clip(idx1, 0, x.size - 1)
    idx2_clipped = np.clip(idx2, 0, x.size - 1)
 
    v1 = np.where((idx1 >= 0) & (idx1 < x.size), x[idx1_clipped], 0.0)
    v2 = np.where((idx2 >= 0) & (idx2 < x.size), x[idx2_clipped], 0.0)
 
    interp = (v1 + v2) / 2.0
 
    out_idx_interp = n[~mask] + INF
    y[out_idx_interp] = interp[~mask]
 
    return y
 

def main():
    img_root = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
    plot(time_scale_signal(signal, 3), title='x[n/3]', saveTo=f'{img_root}/x[n divided by 3].png')
    plot(time_scale_signal(signal, 1), title='x[n/1]', saveTo=f'{img_root}/x[n divided by 1].png')
    plot(time_scale_signal_interpolate(signal, 3), title='x[n/3] with interpolation', saveTo=f'{img_root}/x[n divided by 3]_with_interpolation.png')
    plot(time_scale_signal_interpolate(signal, 1), title='x[n/1] with interpolation', saveTo=f'{img_root}/x[n divided by 1]_with_interpolation.png')

main()
