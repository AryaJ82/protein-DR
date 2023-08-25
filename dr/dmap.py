# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:56:29 2023

@author: Arya
"""


import MDAnalysis as md
from MDAnalysis.analysis import distances
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import kde
from pydiffmap import diffusion_map as dm
import dill


########### Distribution of Rg for first and last 500 

def inter_res_pairwise_euclidean_dis_metric(Ca_ens1: np.array, Ca_ens2: np.array):
    """Euclidean distance between inter-residue C alpha pairwise distance
    matrices.
    
    """
    return np.linalg.norm(Ca_ens1 - Ca_ens2)
    
# data
files = glob.glob(r"/scratch/gradinaru-shared/sic1_trades_rc_ens_20000/*.pdb")
topology = files[0]

u_ens = md.Universe(topology, files, in_memory=True)

# number of alpha carbons = number of residues
N = len(u_ens.residues)
# number of frames in trajectory
M=len(u_ens.trajectory)

# initialize Ca pairwise distance matrix
X = np.zeros((M, int((N*(N-1))/2)))

CA_atoms = u_ens.select_atoms('name CA') # get Ca pairwise distances
i = 0
for frame in u_ens.trajectory: # loop through each frame
    # pairwise Ca distances for a frame
    CA_distances = distances.self_distance_array(CA_atoms.positions)
    X[i, :] = CA_distances # append to array
    i += 1 # increment index
# now, entry i in X contains all Ca pairwise distances for the ith pdb file

n_dc = 20 # number of diffusion components to compute
epsilon = 500
alpha = 0.5
k = 5000


X_stand = (X- np.average(X, axis=0))/np.std(X, axis=0)

mydmap = dm.DiffusionMap.from_sklearn(n_evecs = n_dc, epsilon = epsilon, alpha = alpha, k = k, metric = inter_res_pairwise_euclidean_dis_metric)
dmap = mydmap.fit_transform(X_norm)

evecs = mydmap.evecs
evals = mydmap.evals

num_clusters = 5



# save fitted data for later
# useful if the data needs to be cropped
np.save("sic1-trades-dmap-full-data", dmap)


# save the regular embedding (function from higher to lower dimensional space)
# necessary for backmapping
f = open("sic1-trades-dmap-object.dill", "wb")
dill.dump(mydmap, f)
f.close()



# plot the latent space; pairwise against itself
for i in range(1, 4+1):
    for j in range(i+1, 4+1):
        
        x=dmap[:,i-1]
        y=dmap[:,j-1]
        k = kde.gaussian_kde([x,y])
        density = k.evaluate([x, y])
            
        plt.figure()

        points = plt.scatter(x, y, c=density, cmap="cool")
        cb = plt.colorbar(points)
        cb.set_label("density")

        plt.xlabel(r'$\psi_{i:d}$'.format(i=i))
        plt.ylabel(r'$\psi_{j:d}$'.format(j=j))
        plt.title(f"sic1 TraDes rc ens dmap full; Axis {i} vs {j};\n $\\epsilon$ = {epsilon}, $\\alpha$ = {alpha}, k = {k}")
        
        # save figure as a PNG file
        plt.savefig(f"sic1-TraDes-dmap-full-{i}-{j}.png")


plt.figure()
plt.scatter(range(1,n_dc+1), evals)
plt.xlabel(r'i')
plt.ylabel(r'$\lambda_i$')
plt.title(f"sic1 TraDes dmap full; evals;\n $\\epsilon$ = {epsilon}, $\\alpha$ = {alpha}, k = {k}")
plt.savefig("sic1-trades-dmap-evals-full.png")


