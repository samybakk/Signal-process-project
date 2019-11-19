import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

def norm (signal) :

    #signal : le signal à normaliser


    sig_size = len(signal)
    max_sig = 0
    for i in range(0, sig_size):
        if abs(signal[i]) > max_sig:
            max_sig = abs(signal[i])

    return signal / max_sig

def framing(signal,Fs,step) :

    #signal : le signal qu'on veut frame
    #Fs : fréquence d'échantillonage
    #step : temps (en s) d'une frame


    sig_size = len(signal)
    sig_time = sig_size / Fs
    nbr_frames = sig_time/step
    frames = np.array()
    for i in range(0,round(nbr_frames)) :
        if i != nbr_frames :
            frame = signal[Fs*step*i:Fs*step*(i+1)-1]

        else :
            frame = 0
        frames += frame
if __name__ == '__main__':
    Fs = 16000
    x = np.linspace(0,99,1000)
    signal = 2*np.sin(x)
    signal = norm(signal)

    plt.plot(signal)
    plt.show()