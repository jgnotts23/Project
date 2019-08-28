#!/bin/bash

read -p "Enter directory: " s


# for file in $s/*.WAV
# do
#   sox $file -t raw -
# done |
#   sox -t raw - output.wav



# create an array with all the filer/dir inside ~/myDir
arr=($s/*)

no_files=${#arr[@]}
echo "$no_files wav files found!"

arr_1=("${arr[@]:0:1000}")
arr_2=("${arr[@]:1001:2000}")
arr_3=("${arr[@]:2001:3000}")
arr_4=("${arr[@]:3001:4000}")
arr_5=("${arr[@]:4001}")



#iterate through array using a counter
for ((i=0; i<${#arr_1[@]}; i++));
  do
    sox ${arr[i]} -b 16 -s -c 1 -r 48k -t raw -
  done |
    sox -b 16 -s -c 1 -r 48k -t raw - output.wav




# #iterate through array using a counter
# for ((i=0; i<${#arr_1[@]}; i++)); do
#     #do something to each element of array
#     echo "${arr_1[$i]}"
# done
