#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import os
import glob
import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

for r in glob.glob("./*/*/*.wav"):
	sample_rate, X = scipy.io.wavfile.read(r)
	ceps, mspec, spec = mfcc(X)
	np.save(r, ceps) # cache results so that ML becomes fast