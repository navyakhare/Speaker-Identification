#!/bin/bash -e

for i in $1/*; do
	echo $i
	#cd "$(basename "$i")"
	rm "$i/`(ls $i |tail -1)`"
	#cd ".."
#	(mv $i $(echo $filename | cut -f 1 -d '.').wav)
done


 
	