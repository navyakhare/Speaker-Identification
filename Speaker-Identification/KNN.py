#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import os
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

Y = []
R = []
R_TEST = []
Y_TEST = []
ct = 0
for r in glob.glob("./TRAIN/*/"):
	label = r.split("/")[-2]
	ct += 1
	if ct == 10:
		break
	for fil in glob.glob(r+"*.npy"):
		Y.append(ct)
		X = []
		ceps = np.load(fil)
		num_ceps = len(ceps)
		# X = (np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0, dtype=np.float32))
		X = (np.mean(ceps, axis=0, dtype=np.float32))	
		# X = ceps
		X = X.tolist()
		R.append(X)

ct = 0
for r in glob.glob("./TEST/*/"):
	label = r.split("/")[-2]
	ct += 1
	if ct == 10:
		break
	for fil in glob.glob(r+"*.npy"):
		Y_TEST.append(ct)
		X = []
		ceps = np.load(fil)
		num_ceps = len(ceps)
		# strt = int(num_ceps / 10)
		# end = int(num_ceps * 9 / 10)
		# print ceps[strt:end].shape
		# print np.mean(ceps[strt:end], axis = 0)	
		# print ""
		# X = (np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0, dtype=np.float32))
		X = (np.mean(ceps, axis=0, dtype=np.float32))	
		# X = ceps
		X = X.tolist()
		R_TEST.append(X)


nbrs = RadiusNeighborsClassifier(radius=0.95)
nbrs.fit(R,Y)
Res = nbrs.predict(R_TEST)


nbrs2 = KNeighborsClassifier(n_neighbors=3)
nbrs2.fit(R,Y)
Res2 = nbrs2.predict(R_TEST)

print Y_TEST
print Res
acc = accuracy_score(Y_TEST, Res)
acc2 = accuracy_score(Y_TEST, Res2)
print acc
print acc2
print ''