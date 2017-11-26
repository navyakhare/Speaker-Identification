#!/usr/bin/env bash
for k in *; do
	if [ -d "$k" ]; then
		for f in $k/SX*.wav; do
			echo "mv $f ../TEST/$f"
			break
		done
	fi
done