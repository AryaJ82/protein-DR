# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:46:59 2023

@author: Arya
"""

import MDAnalysis as md
from MDAnalysis.analysis import distances
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm



# ### sic1-a03ws  */.dcd
dcd_path = glob.glob(r'D:/IDP-Data/MD-data-DEShaw/Sic1/DESRES-Trajectory_pnas2018b-sic1-a03ws-protein.tar/DESRES-Trajectory_pnas2018b-sic1-a03ws-protein/pnas2018b-sic1-a03ws-protein/*.dcd')
topology_path = 'D:/IDP-Data/sic1_topology.pdb'
dt = 0.180036 # time step of MD simulation in nanoseconds






u = md.Universe(topology_path, dcd_path, in_memory=True)
M = len(u.trajectory)

ca = u.select_atoms('protein and name CA')
bb = u.select_atoms('protein and backbone')
cterm = ca[0] # c terminus
nterm = ca[-1] # n terminus

Ree = np.zeros(M) # end to end distance
Rg = np.zeros(M) # radius of gyration
i = 0
#Ree = []
for ts in u.trajectory:
    Ree[i] = np.linalg.norm(cterm.position - nterm.position)
    Rg[i] = bb.radius_of_gyration()
    
    i += 1



Y = sm.tsa.acf(Ree, nlags = M)
plt.figure()
plt.title("sic1-a03ws full MD trajectory; End to end distance autocorrelation")
plt.xlabel("Lag Time (ns)")
plt.ylabel("Autocorrelation")
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.scatter( [i*dt for i in range(1,M+1)], Y, s=5)

Y = sm.tsa.acf(Rg, nlags = M)
plt.figure()
plt.title("sic1-a03ws full MD trajectory; Radius of gyration autocorrelation")
plt.xlabel("Lag Time (ns)")
plt.ylabel("Autocorrelation")
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.scatter( [i*dt for i in range(1,M+1)], Y, s=5)


