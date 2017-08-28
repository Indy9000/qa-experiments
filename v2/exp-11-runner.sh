#!/bin/bash
set -e
OUTPUT_FOLDER=$(date "+%Y%m%d-%H%M")
OUTPUT_LOG=$(date "+%Y%m%d-%H%M")
echo $OUTPUT_FOLDER
echo $OUTPUT_LOG
mkdir ./output/$OUTPUT_FOLDER
for i in {1..20}
do
    python exp-11.py 43 3 11 23 31 0.8090 0.0007 56 30| tr -d '\b\r' |sed '/ETA:/d' &> ./output/$OUTPUT_FOLDER/exp-11-output-$(date "+%Y%m%d-%H%M").txt
done
set +e
