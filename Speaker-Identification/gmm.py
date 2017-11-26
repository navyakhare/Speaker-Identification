#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import os
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

Y = []
R = []
R_TEST = []
Y_TEST = []
ct = 0
for r in glob.glob("./TRAIN/*/"):
	label = r.split("/")[-2]
	ct += 1
	for fil in glob.glob(r+"*.npy"):
		Y.append(ct)
		X = []
		ceps = np.load(fil)
		num_ceps = len(ceps)
		X = (np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0, dtype=np.float32))
		# X = ceps
		X = X.tolist()
		R.append(X)

ct = 0
for r in glob.glob("./TEST/*/"):
	label = r.split("/")[-2]
	ct += 1
	for fil in glob.glob(r+"*.npy"):
		Y_TEST.append(ct)
		X = []
		ceps = np.load(fil)
		num_ceps = len(ceps)
		X = (np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0, dtype=np.float32))
		# X = ceps
		X = X.tolist()
		R_TEST.append(X)

R= np.array(R)
R_TEST = np.array(R_TEST)



gmm = GMM(n_components = 8, n_iter = 200, covariance_type='diag',
        n_init = 3)
gmm.fit(R)

gmm_predictions=gmm.predict(R_TEST)
accuracy = accuracy_score(Y_TEST, gmm_predictions)
print accuracy




