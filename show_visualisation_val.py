import numpy as np
import matplotlib.pyplot as plt
import mido
import math
from IPython.display import Audio
from pretty_midi import PrettyMIDI
import time
import fluidsynth
from midi2audio import FluidSynth

import pypianoroll
import music21  # show sheet music
import os

from sklearn.metrics import average_precision_score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    help="The input wav file's path")
args = parser.parse_args()


vls_train = np.load("./" + args.input + ".npy")
print(vls_train.shape)

# vls_train = vls_train.reshape((2, 128, int(vls_train.shape[1]/128)))
# print(vls_train.shape)

#np.save("train_save_prediction", np.array([predictions, labels])) in train
train_val_score = average_precision_score(
    np.array(vls_train[1]).flatten('F'), np.array(vls_train[0]).flatten('F'))
print("train_val_score  : ", train_val_score)

# vls_train.reshape((2, 128, int(vls_train.shape[1]/128)), order='F')
vls = vls_train
#vls = np.load("./" + args.input + ".npy")
print(vls.shape)
# 2, 128, 32906
# 2: 0 = predict, 1 = true vlue, 128 notes, 32906 timestep

vls = np.concatenate([vls, np.zeros([1, *vls.shape[1:]])])
vls = np.transpose(vls, [1, 2, 0])  # (y:128, x:32906, 3:true, predict)
print(vls.shape)

plt.figure(figsize=(10, 4))
plt.imshow(vls[:, :300, 0] > 0.9)  # 0=predict
plt.show()
plt.imshow(vls[:, :300, 0])  # 0=predict
plt.show()
plt.imshow(vls[:, :300, 1])  # 1=true
plt.show()

ground_truth = vls[:, :, 1]  # keep ground truth
prepare_for_midi = vls[:, :, 0]  # pick predict piano-roll
prepare_for_midi = np.transpose(prepare_for_midi, [1, 0])  # 32906,128
print("prepare_for_midi shape: ", prepare_for_midi.shape)
print("prepare_for_midi head:", prepare_for_midi[:3])
print("min: ", prepare_for_midi.min(axis=0))
print("max: ", prepare_for_midi.max(axis=0))
# set 0.9 to be a threvlsidi_binary head:", prepare_for_midi_binary[:3])
print("vls shape", vls[:, :, 0].shape)
prepare_for_midi2 = vls[:, :, 0]
prepare_for_midi_binary2 = np.where(prepare_for_midi > 0.1, 1, 0)
track = pypianoroll.BinaryTrack(pianoroll=prepare_for_midi_binary2)
multi_track = pypianoroll.Multitrack(tracks=[track])
pypianoroll.write("./" + args.input + "_no_post.mid", multi_track)


# apply smoothing (smoothing , sobel smoothing, mean value differntial)######################

plt.figure(figsize=(10, 4))
post_processing = np.transpose(prepare_for_midi > 0.3, [1, 0])
# post_processing = post_processing[:, :300] # view front 300 timestep
print("post_processing.shape", post_processing.shape)  # height = 128, width = N
plt.imshow(post_processing)  # 前 300 step 資料
plt.show()
post_processing = post_processing * 256  # as image

##########################################


##########################################


# sobel kernel
sobel1 = [  # horizonly
    1, 0, -1,
    2, 0, -2,
    1, 0, -1]
sobel2 = [  # vertically
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1]
sobel1_5x5 = [  # vertically
    2, 1, 0, -1, -2,
    2, 1, 0, -1, -2,
    4, 2, 0, -2, -4,
    2, 1, 0, -1, -2,
    2, 1, 0, -1, -2]
sobel2_5x5 = [  # vertically
    2, 2, 4, 2, 2,
    1, 1, 2, 1, 1,
    0, 0, 0, 0, 0,
    -1, -1, -2, -1, -1,
    -2, -2, -4, -2, -2]
# height = 128, width = N

height = post_processing.shape[0]  # 128
width = post_processing.shape[1]  # N
x1 = 0
y1 = 0
x2 = width-1  # img.cols - 1
y2 = height-1  # img.rows - 1
fil = np.zeros(25)  # 5*5 inital

for y in range(y2+1):
    for x in range(x2+1):
        binary_x = 0
        binary_y = 1
        index = 0
        for yy in range(-1, 2):  # -1~1 #sobel filter 3x3 range(-1,2) | range(-2, 3)
            #print("yy:", yy)
            for xx in range(-1, 2):
                if (y + yy >= 0 and y + yy < height):
                    if (x + xx >= 0 and x + xx < width):
                        binary_x += post_processing[y +
                                                    yy][x + xx] * sobel1[index]
                        binary_y += post_processing[y +
                                                    yy][x + xx] * sobel2[index]
                index += 1

        para = math.sqrt((binary_x*binary_x) + (binary_y*binary_y) / 8.0)
        para = 30 - para
        if para < 0:
            para = 0
        for i in range(25):  # 0~24
            fil[i] = para
        fil[2 * 5 + 2] = 30
        bbinary = 0
        fsam = 0
        sadr = 0
        for yy in range(-2, 3):  # -2~3
            #print("yy", yy)
            for xx in range(-2, 3):
                if (y + yy >= 0 and y + yy < height):
                    if (x + xx >= 0 and x + xx < width):
                        bbinary += post_processing[y + yy][x + xx] * fil[sadr]
                        fsam += fil[sadr]
                sadr += 1
        #print("fsam:", fsam)
        if (fsam != 0):
            post_processing[y][x] = bbinary / fsam

post_processing = post_processing / 256
post_processing = np.where(post_processing > 0.9, 1, 0)
plt.imshow(post_processing[:, :300])  # 前 300 step 資料
plt.show()

# Interpolation ########################


def interpolation_array(post_processing, height, width, interpolation_rate):
    for y in range(height):
        for x in range(width):
            accumulation_front = 0
            for i in range(-1*interpolation_rate, 1):  # -interpolation_rate ~ 0
                if (x + i >= 0 and x + i < width):
                    accumulation_front += post_processing[y][x + i]
            accumulation_post = 0
            for i in range(0, interpolation_rate + 1):  # 0 ~ interpolation_rate
                if (x + i >= 0 and x + i < width):
                    accumulation_post += post_processing[y][x + i]
            front_ratio = accumulation_front / float(interpolation_rate)
            post_ratio = accumulation_post / float(interpolation_rate)
            #print("front_ratio: ", front_ratio, " | post_ratio: ", post_ratio)
            if front_ratio != 0.0 and post_ratio != 0.0:
                post_processing[y][x] = 1

            # if accumulation_post / 10
            # = post_processing[y][x]
    return post_processing


height = post_processing.shape[0]  # 128
width = post_processing.shape[1]  # N

post_processing_diff = interpolation_array(
    post_processing, height, width, 5)  # interpolation_rate
print("post_processing_diff.shape", post_processing_diff.shape)
plt.imshow(post_processing_diff[:, :300])  # 前 30post_processing
# original
#original_prediction_front300 = vls[:, :300, 0]
original_prediction = vls[:, :, 0]
print("original_prediction.shape", original_prediction.shape)
# ground_truth_front300 # Logical AND
overlap = original_prediction * ground_truth
union = original_prediction + ground_truth  # ground_truth_front300 # Logical OR

original_IOU = overlap.sum()/float(union.sum())
print("Original IoU:", original_IOU)

# improve
overlap = post_processing_diff * ground_truth  # Logical AND
union = post_processing_diff + ground_truth  # Logical OR

improve_IOU = overlap.sum()/float(union.sum())
print("Improve IoU", improve_IOU)
# post_processing
# 21:33~21:42 :sobel
# 21:42~21:43 :interpolation
# 22:04~22: :sobel
# 22:~22: :interpolation
#######################
# nrow = 1
# ncol = 1
# fig, axs = plt.subplots(nrows=nrow, ncols=nrow)
# axs = np.array(axs)

# pianoroll_vls = pypianoroll.Track(name="piano", pianoroll=prepare_for_midi_binary) #require N, 128
# pianoroll_vls_multi = pypianoroll.Multitrack(tracks=pianoroll_vls)
# plot_return = pypianoroll.plot_multitrack(axs=None, multitrack = pianoroll_vls_multi)
# plt.show()
# vls = np.load("./" + args.input + ".npy")
# post_processing_diff #pianoroll=vls[0].T>0.9
track = pypianoroll.BinaryTrack(pianoroll=post_processing_diff.T)
multi_track = pypianoroll.Multitrack(tracks=[track])
pypianoroll.write("./" + args.input + "_visualisation.mid", multi_track)

# os.environ["musescoreDirectPNGPath"] = "/usr/bin/musescore"
# #environment.set("musescoreDirectPNGPath", "/usr/bin/musescore")
# parsed = music21.converter.parse('predict_track.mid')
# parsed.show()
#######################
# post score
#np.save("train_save_prediction", np.array([predictions, labels])) in train
# train_val_score = average_precision_score(
#     np.array(vls_train[1]), np.array(vls_train[0]))
vls_train = np.load("./" + args.input + ".npy")
print(vls_train.shape)
print("post_processing.shape", post_processing.shape)
train_val_score_post = average_precision_score(
    np.array(vls_train[1]).flatten('F'), np.array(post_processing.flatten('F')))

print("train_val_score  : ", train_val_score)
print("train_val_score_post: ", train_val_score_post)

#######################


# tempo = 139 beats per minute Bach's Prelude in D major
"""
To convert the Tempo Meta-Event's tempo (ie, the 3 bytes that specify the amount of microseconds per quarter note) to BPM:

BPM = 60,000,000/(tt tt tt)

For example, a tempo of 120 BPM = 07 A1 20 microseconds per quarter note.

So why does the MIDI file format use "time per quarter note" instead of "quarter notes per time" to specify its tempo? Well, its easier to specify more precise tempos with the former. With BPM, sometimes you have to deal with fractional tempos (for example, 100.3 BPM) if you want to allow a finer resolution to the tempo. Using microseconds to express tempo offers plenty of resolution.

Also, SMPTE is a time-based protocol (ie, it's based upon seconds, minutes, and hours, rather than a musical tempo). Therefore it's easier to relate the MIDI file's tempo to SMPTE timing if you express it as microseconds. Many musical devices now use SMPTE to sync their playback.
"""
# def arry2mid(ary, tempo=500000):
#     # get the difference
#     new_ary = np.concatenate([np.array([[0] * 128]), np.array(ary)], axis=0) #88
#     changes = new_ary[1:] - new_ary[:-1]
#     # create a midi file with an empty track
#     mid_new = mido.MidiFile()
#     track = mido.MidiTrack()
#     mid_new.tracks.append(track)
#     track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
#     # add difference in the empty track
#     last_time = 0
#     for ch in changes:
#         if set(ch) == {0}:  # no change
#             last_time += 1
#         else:
#             on_notes = np.where(ch > 0)[0]
#             on_notes_vol = ch[on_notes]
#             off_notes = np.where(ch < 0)[0]
#             first_ = True
#             for n, v in zip(on_notes, on_notes_vol):
#                 new_time = last_time if first_ else 0
#                 track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
#                 first_ = False
#             for n in off_notes:
#                 new_time = last_time if first_ else 0
#                 track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
#                 first_ = False
#             last_time = 0
#     return mid_new

# prepare_for_midi = vls[:, :, 0] #pick predict piano-roll
# prepare_for_midi = np.transpose(prepare_for_midi, [1, 0])
# print("prepare_for_midi shape: ", prepare_for_midi.shape)
# print("prepare_for_midi head:", prepare_for_midi[:3])
# #prepare_for_midi = [np.round(element) for element in prepare_for_midi] # float turn to int
# prepare_for_midi = 128 * prepare_for_midi
# prepare_for_midi = prepare_for_midi.astype(int)
# print("prepare_for_midi shape: ", prepare_for_midi.shape)
# print("prepare_for_midi head:", prepare_for_midi[:3])
# mid_new = arry2mid(prepare_for_midi)
# mid_new.save('mid_new.mid')


# sf2_path = './sound_font/MuseScore_General.sf2'  # path to sound font file
#midi_file = './mid_new.mid'

#music = PrettyMIDI(midi_file=midi_file)
#waveform = music.fluidsynth(sf2_path=sf2_path)
#Audio(waveform, rate=44100)

# fs = FluidSynth()
# fs.midi_to_audio('mid_new.mid', 'output.wav')
