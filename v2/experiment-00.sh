#!/bin/bash
set -e
OUTPUT_FOLDER=$(date "+%Y%m%d-%H%M")
OUTPUT_LOG=$(date "+%Y%m%d-%H%M")
echo $OUTPUT_FOLDER
echo $OUTPUT_LOG
mkdir ./output/$OUTPUT_FOLDER
./ea11 20 16 0.1 0.1 ./output/$OUTPUT_FOLDER/ &> ea11-output-$OUTPUT_LOG.log
set +e
