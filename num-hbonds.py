# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:00:23 2023

@author: Arya

Extraction of the number of hydrogen bonds in a trajectory.
Adapted from AshleyNamini's code.
"""

import mdtraj
import numpy as np

file = '.dcd'
topology = '.pdb'
traj = mdtraj.load(file,top=topology)

n_frames = len(traj)


num_hbonds = np.zeros(n_frames)

#iterate through each frame, identify the hbonds and save the indices in hbonds_each_frame
for i in range(n_frames):
    num_hbonds[i] = mdtraj.baker_hubbard(traj[i]).shape[0]