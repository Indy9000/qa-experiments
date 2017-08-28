#!/bin/bash
set -e
OUTPUT_FOLDER=$(date "+%Y%m%d-%H%M")
OUTPUT_LOG=$(date "+%Y%m%d")
echo $OUTPUT_FOLDER
echo $OUTPUT_LOG
mkdir ./output/$OUTPUT_FOLDER
./ea06 50 32 0.25 0.1 ./output/$OUTPUT_FOLDER/ &> ea06-output-$OUTPUT_LOG.log
set +e
