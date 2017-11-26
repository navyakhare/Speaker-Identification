#!/bin/bash -e

for i in $1/*.WAV; do
	echo $i
	mv "$i" "$(basename "$i" .WAV).wav"
#	(mv $i $(echo $filename | cut -f 1 -d '.').wav)
done
	
