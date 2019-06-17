# Split file into 4 second chunks and create a spectrogram for each

import argparse
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
from pylab import *

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
    NFFT = 256
    # Magnitude of the FFT
    mag_frames = np.log(np.absolute(np.fft.rfft(frames, NFFT)))
    mean_filtered_mag_frames = np.array(mag_frames, copy=True)

    results_shape = np.shape(mag_frames)
    for k in range(results_shape[0]):
        for j in range(results_shape[1]):
            row = mag_frames[:,j]
            mean_filtered_mag_frames[k,j] = mag_frames[k,j] - np.mean(row)

    # new_filename = "background_frequency_banks\\"
    new_filename = ""

    # Split filename from structure

    new_filename += args.filename.upper().split('.WAV')[0]
    new_filename += "_" + str(i)
    new_filename += "_spec.png"

    imsave(new_filename, np.flipud(mean_filtered_mag_frames.T), cmap='Greys')

