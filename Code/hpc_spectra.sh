#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=1g
module load anaconda3/personal
module load intel-suite
echo "Python is about to run"
python3  $HOME/spectra.py "$HOME/Cut_SQ283" "$HOME/Spectra_SQ283"
echo "Python has finished running"
