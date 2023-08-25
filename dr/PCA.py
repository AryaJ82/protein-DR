# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:36:50 2023

@author: Arya
"""


import MDAnalysis as md
from MDAnalysis.analysis import distances
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import kde
import sklearn.decomposition

def inter_res_pairwise_euclidean_dis_metric(Ca_ens1: np.array, Ca_ens2: np.array):
    """Euclidean distance between inter-residue C alpha pairwise distance
    matrices.
    
    """
    return np.linalg.norm(Ca_ens1 - Ca_ens2)

# selected MD ensembles at n*selection ns
files = glob.glob(r"/scratch/gradinaru-shared/sic1_trades_rc_ens_20000/*.pdb")
topology = files[0]

u_ens = md.Universe(topology, dcdpath, in_memory=True)

# number of alpha carbons = number of residues
N = len(u_ens.residues)
# number of frames in trajectory
M = len(u_ens.trajectory)

# initialize Ca pairwise distance matrix
X = np.zeros((M, int((N*(N-1))/2)))


CA_atoms = u_ens.select_atoms('name CA') # get Ca pairwise distances
i = 0
for frame in u_ens.trajectory: # loop through each frame
    # pairwise Ca distances for a frame
    CA_distances = distances.self_distance_array(CA_atoms.positions)
    X[i, :] = CA_distances # append to array
    
    i += 1 # increment index


X_stand = (X- np.average(X, axis=0))/np.std(X, axis=0)
PCA = sklearn.decomposition.PCA(n_components=10)
fitted = PCA.fit_transform(X_stand)

# scatter plot of data with density as heat

x= fitted[:,i-1]
y= fitted[:,j-1]
k = kde.gaussian_kde([x,y])
density = k.evaluate([x, y])

plt.figure()
plt.scatter(x , y, c=density, cmap="Reds")
plt.ylabel("PC2")
plt.xlabel("PC1")
plt.title(f"MD sic1 TraDES PCA; Axis 1 vs Axis 2;  heat= density")
plt.colorbar()



# plot evals
plt.figure()
plt.bar(range(1,11), PCA.explained_variance_ratio_[0:10])
plt.ylabel("Explained Variance")
plt.xlabel("Singular Value/Eigenvalue")
plt.title(f"MD sic1 TraDES PCA; Explained variance")