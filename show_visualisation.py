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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    help="The input wav file's path")
parser.add_argument("--visualize",
                    type=int,
                    default=0,
                    help="1. show the output, 0. don't show any output")
parser.add_argument("--output_path",
                    type=str,
                    help="The output midi file's folder")  # "./output_midi/"
args = parser.parse_args()


vls = np.load("./" + args.input + ".npy")
# print(vls.shape)
# 2, 128, 32906
# 2: 0 = predict, 1 = true vlue, 128 notes, 32906 timestep

vls = np.concatenate([vls, np.zeros([1, *vls.shape[1:]])])
vls = np.transpose(vls, [1, 2, 0])  # (y:128, x:32906, 3:true, predict)
# print(vls.shape)

if args.visualize != 0:
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

if args.visualize != 0:
    print("prepare_for_midi shape: ", prepare_for_midi.shape)
    print("prepare_for_midi head:", prepare_for_midi[:3])
    print("min: ", prepare_for_midi.min(axis=0))
    print("max: ", prepare_for_midi.max(axis=0))
    # set 0.9 to be a threshold
    # prepare_for_midi_binary = np.where(prepare_for_midi > 0.9, 1, 0)
    # print("prepare_for_midi_binary shape: ", prepare_for_midi_binary.shape)
    # print("prepare_for_midi_binary head:", prepare_for_midi_binary[:3])
    print("vls shape", vls[:, :, 0].shape)

prepare_for_midi2 = vls[:, :, 0]
prepare_for_midi_binary2 = np.where(prepare_for_midi > 0.1, 1, 0)
track = pypianoroll.BinaryTrack(pianoroll=prepare_for_midi_binary2)
multi_track = pypianoroll.Multitrack(tracks=[track])
pypianoroll.write(args.output_path + args.input +
                  "_no_post.mid", multi_track)


# apply smoothing (smoothing , sobel smoothing, mean value differntial)######################


post_processing = np.transpose(prepare_for_midi > 0.3, [1, 0])
# post_processing = post_processing[:, :300] # view front 300 timestep

if args.visualize != 0:
    # height = 128, width = N
    print("post_processing.shape", post_processing.shape)
    plt.figure(figsize=(10, 4))
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
if args.visualize != 0:
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

if args.visualize != 0:
    print("post_processing_diff.shape", post_processing_diff.shape)
    plt.imshow(post_processing_diff[:, :300])  # 前 300 step 資料
    plt.show()

ground_truth_front300 = ground_truth[:, :300]
# original
#original_prediction_front300 = vls[:, :300, 0]
original_prediction = vls[:, :, 0]
# ground_truth_front300 # Logical AND
overlap = original_prediction * ground_truth
union = original_prediction + ground_truth  # ground_truth_front300 # Logical OR

original_IOU = overlap.sum()/float(union.sum())
if args.visualize != 0:
    print("Original IoU:", original_IOU)

# improve
overlap = post_processing_diff * ground_truth  # Logical AND
union = post_processing_diff + ground_truth  # Logical OR

improve_IOU = overlap.sum()/float(union.sum())
if args.visualize != 0:
    print("Improve IoU", improve_IOU)

vls = np.load("./" + args.input + ".npy")
track = pypianoroll.BinaryTrack(pianoroll=post_processing_diff.T)
multi_track = pypianoroll.Multitrack(tracks=[track])
pypianoroll.write(args.output_path + args.input + "_final.mid", multi_track)
print("&@OK@&")
