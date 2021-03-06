#!/usr/bin/env python2
# -*- coding: UTF-8 -*-


import argparse
import sys
import glob
import os
import itertools
import scipy.io.wavfile as wavfile

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'gui'))
from gui.interface import ModelInterface
from gui.utils import read_wav
from filters.silence import remove_silence

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.

Examples:
    Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
    ./speaker-recognition.py -t enroll -i "./bob/ ./mary/ ./person*" -m model.out

    Predict (predict the speaker of all wav files):
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll" or "predict"',
                       required=True)

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                       required=True)

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       required=True)

    ret = parser.parse_args()
    return ret

#for curr_class in classes:
#>     dirname = os.path.join(data_dir_train, curr_class)
#>     for fname in os.listdir(dirname):
#>         with open(os.path.join(dirname, fname), 'rb') as f:


def task_enroll(input_dirs, output_model):
    m = ModelInterface()
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]
    files = []
    if len(dirs) == 0:
        print "No valid directory found!"
        sys.exit(1)
    for d in dirs:
        label = os.path.basename(d.rstrip('/'))
        print label
        wavs = glob.glob(d + '/*.wav')
        print wavs
        if len(wavs) == 0:
            print "No wav file found in {0}".format(d)
            continue
        print "Label {0} has files {1}".format(label, ','.join(wavs))
        for wav in wavs:
            print wav
           # with open(wav,'r') as f:
                #print f

            fs, signal = read_wav(wav)
            m.enroll(label, fs, signal)
            print"check point1"

    m.train()

    m.dump(output_model)

def task_predict(input_dirs, input_model):
    x=ModelInterface()
    m = ModelInterface.load(input_model)
    total=0
    count=0
    #for f in glob.glob(os.path.expanduser(input_files)):
    #    fs, signal = read_wav(f)
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]
    files = []
    if len(dirs) == 0:
        print "No valid directory found!"
        sys.exit(1)
    for d in dirs:
        label = os.path.basename(d.rstrip('/'))
        wavs = glob.glob(d + '/*.wav')
        
        if len(wavs) == 0:
            print "No wav file found in {0}".format(d)
            continue
        
       # print m.features


        for wav in wavs:
            fs, signal = read_wav(wav)
            label_final = m.predict(fs, signal)
            print wav, '->', label_final
            #print label
            if label_final==label:
                count=count+1
            total=total+1
    
    print "\n"
    accuracy= float(count)/float(total)
    print accuracy

    print "\n"



if __name__ == '__main__':
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model)
    elif task == 'predict':
        task_predict(args.input, args.model)
