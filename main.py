import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
from scipy.signal import argrelextrema, lfilter
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import os
import random
from scikit_talkbox_lpc import lpc_ref



def norm(signal):
    '''
    :param signal: le signal à normaliser
    :return: sig_normed : signal normalisé
    '''

    sig_size = len(signal)
    max_sig = 0
    for i in range(0, sig_size): #on trouve le maximum global du signal
        if abs(signal[i]) > max_sig:
            max_sig = abs(signal[i])
    sig_normed = signal / max_sig #on divise par le maximum global

    return sig_normed


def framing(signal, shifting_step=2500, frames_size=2500,hamming = True):
    '''
   :param signal: le signal qu'on veut frame
   :param shifting_step: la quantité d'échantillons dont on se déplace entre les débuts de chaque frames
   :param frames_size: la taille d'une frame en échantillons
   :return: frames : array des frames
   '''

    sig_size = len(signal)
    frames = []
    i = 0
    while True:
        if (i + frames_size <= sig_size):
            fr_act_size = i + frames_size
        else:
            fr_act_size = sig_size
        frames.append(signal[i:fr_act_size])
        i += shifting_step
        if (i >= sig_size):
            break
    frames = np.array(frames)
    if hamming == True:
        for i in range (0,len(frames)):
            ham = np.hamming(len(frames[i]))
            frames[i] = frames[i] * ham
    return frames


def sig_energy(signal):
    '''
    :param signal: le signal dont on veut calculer l'énergie
    :return: totEnergy : l'énergie du signal
    '''

    totEnergy = 0
    for i in range(0, len(signal)):
        totEnergy += np.power(abs(signal[i]), 2)
    return totEnergy


def pitch(frames,Fs, threshold=10, maxlags=800000, printing=False):
    '''
    :param frames: frames dont on veut déterminer le pitch(fréquence fondamentale)
    :param threshold: Energie à partir de laquelle la frame est voiced
    :param maxlags: décallage max pour la convolution (xcorr/acorr)
    :param printing : Booléen qui détermine si les graphiques sont affichés
    :param hamming : Booléen qui détermine si une hamming Window est appliquée aux frames
    :return: f0: liste des pitch des frames
    '''
    f0 = []
    for i in range(0, len(frames)):

        if sig_energy(frames[i]) > threshold:

            a, b, *_ = plt.acorr(frames[i], maxlags=maxlags) #we only need b, aka the autocorrelation vector

            e = argrelextrema(b, np.greater)  #Local maximum of b, the autocorrelation vector
            loc_max_temp = np.array(e[0]) #temp list
            loc_max = []
            maxt=0
            for h in range(0, len(loc_max_temp)):
                temp = loc_max_temp[h]
                if b[temp] > maxt :
                    loc_max.append(loc_max_temp[h] - maxlags)
                    maxt = b[temp]

            loc_max = np.array(loc_max)
            if len(loc_max) > 1:
                dist = 0
                for j in range(0, len(loc_max) - 1):
                    dist += loc_max[j + 1] - loc_max[j]
                dist = dist / (len(loc_max) - 1)
                tps = dist / Fs
                f0.append(1 / tps)

                if printing==True:
                    plt.subplot(2, 1, 1)
                    plt.plot(frames[i])
                    plt.grid(True)
                    plt.axhline(0, color='black', lw=1)
                    plt.title("frame " + str(i + 1) +'(E = '+str("%.2f" %sig_energy(frames[i]))+')')
                    plt.subplot(2, 1, 2)
                    plt.plot(a, b, 'r-')
                    plt.grid(True)
                    plt.axhline(0, color='black', lw=1)
                    plt.title("fréquence fondamentale : " + str("%.2f" % f0[i]) + "Hz")
                    plt.show()
            else:
                f0.append(0)


        else:
            f0.append(0)
    f0 = np.array(f0)
    return f0

def highPassFilter(signal, preamphaStep=0.67) :
    sig_size = len(signal)
    filteredSig = []
    for i in range(0,sig_size-1) :
        if i>0 :
            filteredSig.append(signal[i]-preamphaStep*signal[i-1])
        else :
            filteredSig.append(signal[i])
    filteredSig = np.array(filteredSig)
    return filteredSig



def formant (frames):

    formanttab = []
    for i in range(0, len(frames)):

        filt_frame = highPassFilter(frames[i])
        temp = lpc_ref(filt_frame, order= 10)
        lpc = np.roots(temp)
        lpc = lpc[np.imag(lpc) >= 0]
        formanttabframe = []
        for j in range (0,len(lpc)) :
            angle = np.arctan2(np.imag(lpc[j]),np.real(lpc[j]))
            freq =angle*(Fs/8*np.pi)

            if (freq<20000 and freq>500):
                formanttabframe.append(freq)
                formanttabframe.sort()
        formanttab.append((formanttabframe))

    formanttab = np.array(formanttab)
    formanttab = np.sort(formanttab)

    return formanttab
if __name__ == '__main__':

    pathM = "C://Users//frost//Documents//BA3//signal processing//sig//male//"
    pathF = "C://Users//frost//Documents//BA3//signal processing//sig//female//"

    filesM = []
    filesF = []
    for i in range(2) :
        file = random.choice(os.listdir(pathM))

        Fs, rawfileM = read(pathM+file)
        Fs2, rawfileF = read(pathF+file)

        filesM.append(np.array(rawfileM, dtype=float))
        filesF.append(np.array(rawfileF, dtype=float))

    filesM = np.array(filesM)
    filesF = np.array(filesF)

    filesMNormed = []
    filesFNormed = []
    framesMtemp =[]
    framesFtemp =[]
    framesM = []
    framesF = []
    pitchM = []
    pitchF = []
    formantM = []
    formantF = []
    resultM = []
    resultF = []
    for i in range(len(filesM)) :
        filesMNormed.append(norm(filesM[i]))
        filesFNormed.append(norm(filesF[i]))

        framesMtemp.append(framing(filesMNormed[i],round(Fs/100),round(Fs/30),hamming= True))
        framesFtemp.append(framing(filesFNormed[i],round(Fs/100),round(Fs/30),hamming= True))
        framesM.append(framesMtemp[0])
        framesF.append(framesFtemp[0])

        pitchM.append(pitch(framesM[i],Fs,maxlags = round(Fs/50)))
        pitchF.append(pitch(framesF[i],Fs,maxlags = round(Fs/50)))

        #formantM.append(formant(framesM[i]))
        #formantF.append(formant(framesF[i]))

    resultM = [filesMNormed,framesM,pitchM]
    resultF = [filesFNormed,framesF,pitchF]

    moypitchM = 0
    moypitchF = 0
    M =0
    F =0
    for j in range (len(pitchM[0])) :
        if pitchM[0][j] != 0:
            moypitchM += pitchM[0][j]
            M = M+1
            print(M)
        if pitchF[0][j] != 0:
            moypitchF += pitchF[0][j]
            F = F+1
    moypitchM = moypitchM/M
    moypitchF = moypitchF/F
    print('male :',moypitchM)
    print('female :',moypitchF)





