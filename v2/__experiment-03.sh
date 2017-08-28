#!/bin/bash
set -e
OUTPUT_FOLDER=$(date "+%Y%m%d-%H%M")
OUTPUT_LOG=$(date "+%Y%m%d-%H%M")
echo $OUTPUT_FOLDER
echo $OUTPUT_LOG
mkdir ./output/$OUTPUT_FOLDER
./ea09 10 32 0.1 0.25 ./output/$OUTPUT_FOLDER/ &> ea09-output-$OUTPUT_LOG.log
set +e
