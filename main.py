import matplotlib.pyplot as plt
import numpy as np
import scipy as sc



if __name__ == '__main__':

    x = np.linspace(0,99,100)
    sin = 2*sc.sin(x)
    signal = sin
    max_sig = 0
    plt.subplot(signal)
    for i in signal :
        if signal[i]> max_sig :
            max_sig = signal[i]

    signal = signal/max_sig
    plt.subplot(signal)
    plt.show()