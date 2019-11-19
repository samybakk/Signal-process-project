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

def framing(signal,shifting_step=16000,frames_size = 16000) :

    #signal : le signal qu'on veut frame
    #shifting_step : la quantité d'échantillons dont on se déplace entre les débuts de chaque frames
    #frames_size : la taille d'une frame en échantillons

    sig_size = len(signal)
    frames = []
    i=0
    while True :
        if(i+frames_size <= sig_size):
            fr_act_size = i+frames_size
        else:
            fr_act_size = sig_size
        frames.append(signal[i:fr_act_size])
        i+=shifting_step
        if(i>sig_size):
            break
    frames = np.array((frames))
    return frames

if __name__ == '__main__':
    Fs = 1000
    x = np.linspace(0,9999,10000)
    signal = 2*np.sin(x/1000)
    signal = norm(signal)
    frames= framing(signal,shifting_step=900,frames_size= Fs)

    for i in range (0,5) :
        plt.plot(frames[i])
        plt.show()
