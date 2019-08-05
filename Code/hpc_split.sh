#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=1g
module load anaconda3/personal
module load intel-suite
echo "Python is about to run"
python3  $HOME/split_wav.py "$HOME/SQ283" "$HOME/Cut"
echo "Python has finished running"
