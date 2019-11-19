import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

def norm (signal) :
    sig_size = len(signal)
    max_sig = 0
    for i in range(0, sig_size):
        if abs(signal[i]) > max_sig:
            max_sig = abs(signal[i])

    return signal / max_sig


if __name__ == '__main__':

    x = np.linspace(0,99,1000)
    sin = 2*sc.sin(x)
    signal = sin
    signal = norm(signal)

    plt.plot(signal)
    plt.show()