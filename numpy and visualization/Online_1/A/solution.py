import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """

    #return (np.interp(t_query,t_original,x_original))


    ans=t_query.copy()
    for i in range(t_query.size):
        l=T_MIN-1
        r=T_MAX+1
        lv=-1
        rv=-1
        for m in range(x_original.size):
            if(t_query[i]>t_original[m] and t_original[m]>l):
                l=t_original[m]
                lv=x_original[m]
            if(t_query[i]<t_original[m] and (t_original[m]<r)):
                r=t_original[m]
                rv=x_original[m]
            if(t_query[i]==t_original[m]):
                l=r=t_original[m]
                lv=rv=x_original[m]
                continue
        ans[i]=(lv+rv)/2
        if(r==T_MAX+1): ans[i]=lv
        if(l==T_MIN-1 and r!=T_MAX+1): ans[i]=rv

    return ans
    # raise NotImplementedError


def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """

    return (interpolate_signal(t,x,t/k))

    #raise NotImplementedError


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """

    fig = plt.figure(figsize=(10, 6), dpi=100)
    fig.suptitle(title, fontsize=16)


    ax1 = fig.add_subplot(111)

    ax1.plot(t,x, label='x(t) vs t')
    ax1.grid(True)

    ax1.plot(t,y, label=f'y(t)=x(t)/2 vs t')

    ax1.legend()

    plt.show()
    # raise NotImplementedError


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()
