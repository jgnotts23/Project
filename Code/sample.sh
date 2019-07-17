#!/bin/bash

box_path = "/run/user/1000/gvfs/dav:host=dav.box.com,ssl=true/dav/"

read -p "Enter box file name: " box_file

ls |sort -R |tail -$N |while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
done
