# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:37:01 2023

@author: Arya
"""




import MDAnalysis as md
from MDAnalysis.analysis import distances
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import kde
from sklearn.manifold import TSNE
import dill


########### Distribution of Rg for first and last 500 

def inter_res_pairwise_euclidean_dis_metric(Ca_ens1: np.array, Ca_ens2: np.array):
    """Euclidean distance between inter-residue C alpha pairwise distance
    matrices.
    
    """
    return np.linalg.norm(Ca_ens1 - Ca_ens2)
    

files = "/scratch/gradinaru-shared/drknsh3-all-structs/all_structs_cat.pdb"
topology = "/scratch/jafari24/IDP-Data/topology/drknSH3-topology-alt.pdb"

u_ens = md.Universe(topology, files, in_memory=True)

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
# now, entry i in X contains all Ca pairwise distances for the ith pdb file


n_dc = 10 # number of diffusion components to compute
perplexity = 300.0
n_iter = 5000
init = 'pca'
learning_rate = 200


# standardize data componentwise
X_stand = (X- np.average(X, axis=0))/np.std(X, axis=0)

mytsne = TSNE(n_components=n_dc, perplexity= perplexity, learning_rate=learning_rate, n_iter=n_iter,
             metric=inter_res_pairwise_euclidean_dis_metric, method='exact', init=init)

fitted = mytsne.fit_transform(X_stand)



# save fitted data for later
# useful if the data needs to be cropped
np.save("adk-tsne-data.npy", fitted)

# save the regular embedding (function from higher to lower dimensional space)
# necessary for backmapping
f = open("adk-tsne-object", "wb")
dill.dump(mytsne, f)
f.close()


# plot the latent space
for i in range(1, 4+1):
    for j in range(i+1, 4+1):
        
        x=fitted[:,i-1]
        y=fitted[:,j-1]
        k = kde.gaussian_kde([x,y])
        density = k.evaluate([x, y])
            
        plt.figure()

        points = plt.scatter(x, y, c=density, cmap="cool")
        cb = plt.colorbar(points)
        cb.set_label("density")

        plt.xlabel(r'$\psi_{i:d}$'.format(i=i))
        plt.ylabel(r'$\psi_{j:d}$'.format(j=j))
        plt.title(f"drkNSH3 folding-unfolding transition; TraDes truncated last 10000 TSNE; Axis {i} vs {j};\n perplexity = {perplexity}, n_iter = {n_iter}, \learning rate = {learning_rate},\n init = {init} , metrc=inter-residue distance")
        
        # save figure as a PNG file
        plt.savefig(f"adk-tsne-{i}-{j}.png")


