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

#print R.shape
#print R_TEST.shape

#R_TEST = np.array(R_TEST).reshape((1, -1))
#R = np.array(R).reshape((1, -1))
svm_model_linear = SVC(kernel = 'poly', C = 2).fit(R, Y)
svm_predictions = svm_model_linear.predict(R_TEST)

#print svm_predictions.shape
# print svm_predictions
#svm_predictions = np.array(svm_predictions).reshape((1, -1))
accuracy = accuracy_score(Y_TEST, svm_predictions)
print accuracy
# cm = confusion_matrix(Y_TEST, svm_predictions)
# print cm

# print accuracy

#nbrs = KNeighborsClassifier(n_neighbors=5)
#nbrs.fit(R,Y)
#Res = nbrs.predict(R_TEST)

#acc = accuracy_score(Y_TEST, Res)
#print acc