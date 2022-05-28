import argparse  # parser
import os
from os import path
import sys
import subprocess
import re  # for read log file and string processing
import datetime
from tabulate import tabulate
import pandas as pd
import numpy as np
import threading  # for cpu monitor
import time
from pathlib import Path

# Execute this file:
# python python_auto_execute.py --input Prelude_No_5_BWV_850_in_D_Major --denoise 0


def parser():
    parser = argparse.ArgumentParser(
        description="python auto execute")
    parser.add_argument("--input",
                        type = str,
                        default = "Prelude_No_5_BWV_850_in_D_Major",
                        help = "given string of input filename of wav file")
    parser.add_argument("--denoise",
                        type = int,
                        default = 0,
                        help = "If your audio file is record from real environment then give 1 to denoising.")
    parser.add_argument("--output_path",
                       type=str,
                       help="The output midi file's folder") #"./output_midi/"
    return parser.parse_args()


def striplist(l):
    return([x.strip() for x in l])


def split(word):
    return [char for char in word]


def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * float(x)
    return result


start = time.time()
args = parser()
print(args)
input_filename =  str(args.input)
print(input_filename)
# keep naming unique ===============================================================
date_of_current = datetime.datetime.now()
current_date_inside = str(date_of_current.year) + "-" + str(date_of_current.month) + "-" + str(date_of_current.day) + \
    "-" + str(date_of_current.hour) + "-" + \
    str(date_of_current.minute) + "-" + str(date_of_current.second)
print(current_date_inside)

# denoising and get tempo ==========================================================
# python execute_all_preprocessing.py --input <filename>
if int(args.denoise) != 0:
    cmd = 'python execute_all_preprocessing.py ' + \
            '--input ' + input_filename
    process1 = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    stdout, stderr = process1.communicate()
    exit_code = process1.wait()
    print("*stdout, stderr, exit_code:\n", stdout, stderr, exit_code)
    # correct = print("&@OK@&")
    result_of_table = re.findall(
        '\&\@(.*?)\@\&', stdout, re.DOTALL)
    print("result_of_table1:", result_of_table)

# wav->npy =========================================================================
process2 = None
if int(args.denoise) == 0:
    # python make_dataset.py --input piano_ver_Pneumatic
    denoise_cmd = 'python make_dataset.py ' + \
        '--input ' + input_filename
    process2 = subprocess.Popen(
        denoise_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
else:
    denoise_cmd = 'python make_dataset.py ' + \
        '--input ' + input_filename + '_reduction'
    process2 = subprocess.Popen(
        denoise_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

stdout2, stderr2 = process2.communicate()
exit_code2 = process2.wait()
print("*stdout2, stderr2, exit_code2:\n", stdout2, stderr2, exit_code2)

result_of_table2 = re.findall(
    '\&\@(.*?)\@\&', stdout2, re.DOTALL)
print("result_of_table2:", result_of_table2)

# output visualisation.npy =========================================================
process3 = None
if int(args.denoise) == 0:
    # python visualiser.py --input piano_ver_Pneumatic_reduction
    visualiser_cmd = 'python visualiser.py ' + \
        '--input ' + input_filename
    process3 = subprocess.Popen(
        visualiser_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
else:
    visualiser_cmd = 'python visualiser.py ' + \
        '--input ' + input_filename + '_reduction'
    process3 = subprocess.Popen(
        visualiser_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

stdout3, stderr3 = process3.communicate()
exit_code3 = process3.wait()
print("*stdout3, stderr3, exit_code3:\n", stdout3, stderr3, exit_code3)

result_of_table3 = re.findall(
    '\&\@(.*?)\@\&', stdout3, re.DOTALL)
print("result_of_table3:", result_of_table3)

# final result output ==============================================================
process4 = None
if int(args.denoise) == 0:
    # python show_visualisation.py --input piano_ver_Pneumatic_reduction_visualisation
    show_visualisation_cmd = 'python show_visualisation.py ' + \
        '--input ' + input_filename + '_visualisation'  + ' --output_path ' + args.output_path
    process4 = subprocess.Popen(
        show_visualisation_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
else:
    show_visualisation_cmd = 'python show_visualisation.py ' + \
        '--input ' + input_filename + '_reduction_visualisation' + ' --output_path ' + args.output_path
    process4 = subprocess.Popen(
        show_visualisation_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

stdout4, stderr4 = process3.communicate()
exit_code4 = process4.wait()
print("*stdout4, stderr4, exit_code4:\n", stdout4, stderr4, exit_code4)

result_of_table4 = re.findall(
    '\&\@(.*?)\@\&', stdout4, re.DOTALL)
print("result_of_table4:", result_of_table4)
# output: <args.input>_reduction_visualisation_final

# Check output file whether exists =================================================

end = time.time()
print("Spend time: ", format(end-start))

if int(args.denoise) == 0:
    my_file = Path(args.output_path + input_filename + "_visualisation_final.mid")
    if my_file.is_file():
        print("Final output exist!")
    else:
        print("Final output not exist...")
else:
    my_file = Path(args.output_path + input_filename + "_reduction_visualisation_final.mid")
    if my_file.is_file():
        print("Final output exist!")
    else:
        print("Final output not exist...")


############################################################################################
