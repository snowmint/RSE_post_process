import os
import subprocess
import argparse
import numpy as np

from scipy.io import wavfile
from intervaltree import IntervalTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mido

# from resample_musicnet import resample_musicnet# resample_musicnet

# import noise_reduction as nr
# import input_transcript as it
# import bpm_detector as bd

# python execute_all_preprocessing.py --input ./piano_ver_Pneumatic.wav

parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    help="The input wav filename")
parser.add_argument("--visualize",
                    type=int,
                    default=0,
                    help="1. show the output, 0. don't show any output")
# parser.add_argument("--output",
#                     type=str,
#                     help="The output noise reduction wav file's path")
args = parser.parse_args()

#filename = args.input.split('.')[-2]
#filename = filename.strip('/')
filename = args.input
if args.visualize != 0:
    print("Filename: ", filename)
    print("\n==== BPM detector ====================")

pid_get_bpm = subprocess.Popen(
    ["python", "bpm_detector.py", "--filename", "./" + args.input + ".wav"], stdout=subprocess.PIPE)
terminal_out_bpm, err1 = pid_get_bpm.communicate()
if args.visualize != 0:
    print("bpm detector process:", pid_get_bpm, " || Any error? ", err1)
    terminal_out_bpm = np.round(
        float(terminal_out_bpm.split()[-1]))  # least integer

if args.visualize != 0:
    print(terminal_out_bpm)
    print("\n==== Noise Reduction =================")

noise_reduction_output_path = "./" + filename + "_reduction"
pid_noise_reduction = subprocess.Popen(["python", "noise_reduction.py", "--input", "./" +
                                       args.input + ".wav", "--output", noise_reduction_output_path + ".wav"], stdout=subprocess.PIPE)
terminal_out_noise_reduction, err2 = pid_noise_reduction.communicate()
if args.visualize != 0:
    print("bpm detector process:", pid_noise_reduction, " || Any error? ", err2)
    print(terminal_out_noise_reduction)

print("&@OK@&")
