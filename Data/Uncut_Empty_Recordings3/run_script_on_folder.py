import sys
import os

script_name = ""
folder = ""
arguments = []
if(len(sys.argv) >= 3):
    script_name = str(sys.argv[1])
    folder = str(sys.argv[2])
    arguments = sys.argv[3:]
else:
    print("run_script_on_folder.py script_name folder_name [arg_0, arg_1, ...]")
    exit()

for file in os.listdir(folder):
    if file.upper().endswith(".WAV"):
        file_location = folder + "/" + file
        command = "python %s %s" % (script_name, file_location)
        for i in range(len(arguments)):
            command += " " + arguments[i]
        # print(command)
        os.system(command)
        # print("-")
