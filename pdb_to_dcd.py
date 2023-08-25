# Author: Thomas Tsangaris

# Purpose: To create file of dcd files that are the equivalent representation 
# of the inputted folder of pdb files.

# Sample use in linux terminal: python3 pdb_to_dcd.py /full_path_to_ensemble_folder/

import os
import sys
import glob
import subprocess
import time

start_time = time.time() # start time

dir_path = sys.argv[1] # path to folder of pdb files
pdb_files = glob.glob(dir_path + '*.pdb') # list of pdb files
results_path = dir_path + 'dcd_Files/' # directory where the results will be placed

if not os.path.exists(results_path): # if the results folder does not exist
    os.makedirs(results_path) # create the results folder
os.chdir(results_path) # change current working directory, honestly we don't need to do this,
# but probably good practice to do so when working with os library

print('Running mdconvert from mdtraj on ' + str(len(pdb_files)) + ' pdb files')

i = 0 # resulting dcd files will be numbered in an ascending manner starting from zero
for pdb in pdb_files: # loop through each pdb file
    print(pdb)
    bash_cmd = ['mdconvert', pdb, '-o', results_path + 'pdb_to_dcd_' + str(i) + '.dcd'] # bash command with arguments
    process = subprocess.run(bash_cmd, stdout = subprocess.PIPE) # run mdconvert
    i += 1

print('mdconvert calculations completed')
print("Program took %s seconds" % (time.time() - start_time))
