#!/bin/bash

read -p "Enter directory: " s

find $s -size  0 -print0 |xargs -0 rm
