# Split file into 4 second chunks and create a spectrogram for each

import argparse
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
from pylab import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="The name of the recording", type=str)
args = parser.parse_args()

sample_rate, signal = scipy.io.wavfile.read(args.filename)

signal_chunks = int(len(signal) / (4 * sample_rate))

for i in range(signal_chunks):
    start = i * 4
    end = start + 4

    cut_signal = signal[start*sample_rate:int(end * sample_rate)]

    # Pre-emphasis filter
    pre_emphasis = 0.97
    cut_signal = np.append(cut_signal[0], cut_signal[1:] - pre_emphasis * cut_signal[:-1])

    # Framing
    frame_length_ms = 0.1
    frame_step_ms = 0.01

    frame_length = frame_length_ms * sample_rate
    frame_step = frame_step_ms * sample_rate

    signal_length = len(cut_signal)

    # Split into frames
    # ------------------------------
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure there is at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(cut_signal, z)

    # Create array with width=frame_step, each value in a row equalling the row_number*step_length
    stepped_array = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    # Create an array of indices mapping out overlapping frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + stepped_array

    # Use indices to collect overlapping frames
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply Hamming filter
    frames *= np.hamming(frame_length)
    # ------------------------------

    # N-point FFT
    NFFT = 512
    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Apply filter banks
    nfilt = 40

    # Lower and upper bounds of the frequency range
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

    # Equally space points in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    # Convert Mel points to Hz
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)

    # If a value in filter_banks == 0, set it to the smallest possible positive float instead
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

    # Convert to dB
    filter_banks = 20 * np.log10(filter_banks)

    # MFCC
    num_ceps = 12

    # Keep 2-13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

    # Mean normalisation
    # filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    new_filename = "background_mfcc\\"
    new_filename += os.path.basename(args.filename).upper().split('.WAV')[0]
    new_filename += "_" + str(i)
    new_filename += "_mfcc.png"

    imsave(new_filename, mfcc.T, cmap=cm.Greys)

