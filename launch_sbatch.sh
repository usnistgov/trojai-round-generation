#!/bin/bash

# find the directory with largest value
MODEL_DIR=/mnt/isgnas/home/mmajursk/trojai/r9/models
NJOBS=20
MODELS_PER_JOB=10

A=($MODEL_DIR/id-*)
HIGHEST_DIR="${A[-1]##*/}"

HIGHEST_VAL=$(echo $HIGHEST_DIR | tr -dc '0-9')
TRIM_HIGHEST=$(echo $HIGHEST_VAL | sed 's/^0*//')

START_INDEX="$((TRIM_HIGHEST + 1))"

for i in $(seq 1 $NJOBS);
do
  START_RUN=$((START_INDEX + (i-1) * $MODELS_PER_JOB))
  echo "Executing: $i with start: $START_RUN with length: $MODELS_PER_JOB"
  sbatch sbatch_script.sh $START_RUN $MODELS_PER_JOB
done









