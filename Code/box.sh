#!/bin/bash

read -p "Enter box file name: " s
read -p "Enter local file path: " t

echo "Beginning data transfer..."

rclone copy -P iubox:/$s $t

echo "Transfer complete!"