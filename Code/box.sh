#!/bin/bash

read -p "Enter box file name: " s
read -p "Enter local file path: " t

echo "Beginning data transfer..."

rclone copy -P --ignore-existing iubox:/$s $t

echo "Transfer complete!"

#/media/hpc/jsg18/ephemeral
#rclone copy -P "/run/user/1000/gvfs/dav:host=dav.box.com,ssl=true/dav/SQ258/Audio 2" "/media/jgnotts23/Samsung_external"