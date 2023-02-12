from scipy.io import wavfile
import glob
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy import signal
import sys

filename = '/datasets1/LRS3/audio/trainval/zXCiv4sc5eY/50001.wav'


def main():
    try:
        fs, data = wavfile.read(filename)

        if len(data.shape) != 1:
            data = data[:, 0]
        data = data[int(len(data) * 0.1): int(len(data) * 0.9)]  # cut ends of the signal to avoid interference

        # keiser
        window_array = np.kaiser(len(data), 500)
        signal_kaiser = window_array * data

        # FFT
        signal1 = fft(signal_kaiser)
        signal1 = abs(signal1) / len(data) * 2
        signal1[0] = 0

        n = len(data)
        freqs = [fq * fs / n for fq in range(int(n))]

        N = 10000
        data = signal1[:N]
        freqs = freqs[:N]

        plt.plot(freqs, data)
        plt.show()

        # signal.decimate
        kopia_fft = data.copy()
        for k in range(2, 6):
            d = signal.decimate(data, int(k))
            kopia_fft[:len(d)] *= d

        maxIndex = np.argmax(kopia_fft)
        topFreq = freqs[maxIndex]
        print(f"{'M' if topFreq < 170 else 'F'}")

    except:
        print("M")
        return 1


main()
