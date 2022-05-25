import pypianoroll
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    help="The input wav file's path")
parser.add_argument("--visualize",
                    type=int,
                    default=0,
                    help="1. show the output, 0. don't show any output")
# parser.add_argument("--output",
#                     type=str,
#                     help="The output noise reduction wav file's path")
args = parser.parse_args()

input_midi_file = None
input_audio_file = "./" + args.input + ".wav"  # "./test_input.wav"
output_file_name = "./" + args.input + ".npy"  # "./test_input.npy"

fs = 11000  # samples/second
d = 8192  # number of features
stride = 128  # stride in test set


# read wave file
samplerate, wave = wavfile.read(input_audio_file)
if args.visualize != 0:
    print("input sample rate:", samplerate)
    print("input wave shape:", wave.shape)

if (len(wave.shape) > 1):
    if args.visualize != 0:
        print("stereo")
    wave = wave[:, 1]  # pick left one, get right one also be fine

# down sample wave
wave_ds = signal.resample(wave, wave.shape[0]*fs//samplerate)  # wave[:, 0]
wave_ds = np.concatenate([np.zeros(d//2), wave_ds])


# create input x
first_row = np.arange(d)[np.newaxis, :]
time_offset = np.arange((wave_ds.shape[0]-d)//stride)[:, np.newaxis]*stride
input_x = wave_ds[first_row+time_offset]
# print("input_x shape:", input_x.shape)

# read midi file
if input_midi_file:
    multitrack = pypianoroll.read(input_midi_file)
    track = multitrack.tracks[0]
    if args.visualize != 0:
        print("midi resolution:", multitrack.resolution)
        print("midi tempo:", multitrack.tempo[0])
        print("midi track shape:", track.pianoroll.shape)
    plt.imshow(track.pianoroll.T)

    # time scale for midi file to match input data
    id = np.arange(track.pianoroll.shape[0])
    t = np.cumsum(60/multitrack.tempo/multitrack.resolution)
    t = t-t[0]
    t_new = np.arange(input_x.shape[0]) * stride / fs
    id_new = np.floor(np.interp(t_new, t, id)).astype(int)

    # resample track
    track_rs = track.pianoroll[id_new]
    track_rs = (track_rs > 1)+1

    # create input y
    input_y = track_rs
    while input_y.shape[1] + track_rs.shape[1] < input_x.shape[1]:
        input_y = np.concatenate([input_y, track_rs], 1)

    input_y = np.concatenate([
        input_y,
        np.zeros([input_y.shape[0], input_x.shape[1]-input_y.shape[1]])
    ], 1)

else:
    input_y = np.zeros(input_x.shape)

if args.visualize != 0:
    print("input_y shape:", input_y.shape)


dataset = np.concatenate([
    input_x[:, np.newaxis],
    input_y[:, np.newaxis]
], 1)

np.save(output_file_name, dataset)

print("&@OK@&")
