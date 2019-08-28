#!/usr/bin/env python3

"""  """

__appname__ = 'sample.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


import os

myPath = "/run/user/1000/gvfs/dav:host=dav.box.com,ssl=true/dav/SQ258"
mySize = 1000000 #in bytes = 1mb
#os.chdir(myPath)

filesList= []
filesList_nonzero = []
i = 0

for path, subdirs, files in os.walk(myPath):
    for name in files:
        if i % 100 == 0:
            filesList.append(os.path.join(path, name))
            print(i)
            i = i + 1
        else:
            i = i + 1
            continue

for i in filesList:
    # Getting the size in a variable
    fileSize = os.path.getsize(str(i))

    # Print the files that meet the condition
    if int(fileSize) >= int(mySize):
        i = i + 1
        print(i)
        filesList_nonzero.append(os.path.join(path, name))

        #print "The File: " + str(i) + " is: " + str(fileSize) + " Bytes"


