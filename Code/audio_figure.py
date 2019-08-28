#!/usr/bin/env python3

"""  """

__appname__ = 'audio_figure.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('../Data/Gunshot_recordings/Cut_Gunshot_Recordings/Cut_Gunshot_Recordings1/002/58FBD3C1.wav')
times = np.arange(len(data))/float(samplerate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(10, 6))
plt.fill_between(times, data, color='k')
plt.xlim(times[8000], times[17500])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig('../Results/time_amplitude.png', dpi=100)
plt.close()




plt.interactive(False)

# Load audio data
clip, sample_rate = librosa.load('../Data/Gunshot_recordings/Cut_Gunshot_Recordings/Cut_Gunshot_Recordings1/002/58FBD3C1.wav', sr=None)

# Plot spectrogram
fig = plt.figure(figsize=[0.72, 0.72])
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
librosa.display.specshow(
    librosa.power_to_db(S, ref=np.max))


plt.savefig('../Results/example_spectra.png', dpi=400, bbox_inches='tight', pad_inches=0)
plt.close()
fig.clf()
plt.close(fig)
plt.close('all')





y, sr = librosa.load('../Data/Gunshot_recordings/Cut_Gunshot_Recordings/Cut_Gunshot_Recordings1/002/58FBD3C1.wav', sr=None)
librosa.feature.melspectrogram(y=y, sr=sr)
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=258, fmax=8000)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.ylim([0, 4096])
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig('../Results/detailed_spectra.png', dpi=400, bbox_inches='tight', pad_inches=0)
plt.close()
