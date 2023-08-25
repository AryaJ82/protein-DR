# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:09:51 2023

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

from scipy.optimize import minimize
from sklearn.manifold import MDS

########### Distribution of Rg for first and last 500 

def rotate_vector(v1, v2):
    """ return the matrix which rotates the vector v1 to v2
    """
    
    axis = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    axis_normalized = axis / np.linalg.norm(axis)

    S = np.array([[0, -axis_normalized[2], axis_normalized[1]],
                  [axis_normalized[2], 0, -axis_normalized[0]],
                  [-axis_normalized[1], axis_normalized[0], 0]])

    theta = np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    R = np.eye(3) + np.sin(theta) * S + (1 - np.cos(theta)) * np.dot(S, S)

    return R


def MDS_method(distances):
    """ Set of coordinates in 3D space which give the pariwise distance in distances
    """
    dissimilarity_matrix = np.zeros((N, N))
    dissimilarity_matrix[np.triu_indices(N, 1)] = distances 
    dissimilarity_matrix = dissimilarity_matrix + dissimilarity_matrix.T

    mds = MDS(n_components=3, dissimilarity="precomputed")
    return mds.fit_transform(dissimilarity_matrix)

def pdb_from_interresidue(u_ens, CA_atoms, X, name):
    """universe object, C alpha atom group, inter residue distances, name of pdb
    generate a pdb file from a given set of inter-residue distances
    """
    coordinates = MDS_method(X)
    coordinates = coordinates - coordinates[0]

    v = CA_atoms[1].position - CA_atoms[0].position
    w = coordinates[1] - coordinates[0]
    rotation_matrix = rotate_vector(w, v)
    coordinates = np.dot(rotation_matrix, coordinates.T).T


    residues = [res.resname for res in u_ens.residues]
    with open(f"{name}.pdb", "w") as f:
        for i in range(N):
            x = str(round(coordinates[i][0], 3)).rjust(12, " ")
            y = str(round(coordinates[i][1], 3)).rjust(8, " ")
            z = str(round(coordinates[i][2], 3)).rjust(8, " ")
            f.write(f"ATOM{str(CA_atoms[i].id).rjust(7, ' ')}  CA  {residues[i]} A{str(i+1).rjust( 4, ' ')}{x}{y}{z}  1.00  0.00      A    C  \n")
        f.write("TER")



def inter_res_pairwise_euclidean_dis_metric(Ca_ens1: np.array, Ca_ens2: np.array):
    """Euclidean distance between inter-residue C alpha pairwise distance
    matrices.
    
    """
    return np.linalg.norm(Ca_ens1 - Ca_ens2)
    

# data
files = "/scratch/gradinaru-shared/MDAnalysis-ADK/adk_DIMS_transition_trajectory.dcd"
topology = "/scratch/gradinaru-shared/MDAnalysis-ADK/adk_DIMS_transition_topology.pdb"


u_ens = md.Universe(topology, files, in_memory=True)
CA_atoms = u_ens.select_atoms('name CA') # get Ca pairwise distances

# number of alpha carbons = number of residues
N = len(u_ens.residues)
# number of frames in trajectory
M = len(u_ens.trajectory)

###############################################################################
## Below can be replaced with 
# mydmap = dill.load("adk-dmap-object.dill")
# fitted = np.load("adk-dmap-data.npy")

# initialize Ca pairwise distance matrix
X = np.zeros((M, int((N*(N-1))/2)))

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
fitted = mydmap.fit_transform(X_norm)

###############################################################################




# set of points 
num_steps = 8
l =  -14.457
u =  14.457
st = (u - l) / num_steps
i = 0

initial_guess = X_stand[10] # just some point in the set
p = np.zeros(n_dc)
p[0] = l

v = np.array([1, 0, 0])
while (i <= num_steps):
    # Latent Space to inter-residue space
    distances = minimize(lambda Y: np.linalg.norm(mydmap.transform(Y) - p), initial_guess).x
    
    # feeding the previous result as a guess for the next point will keep conformers consistent
    initial_guess = distances.copy()
    
    # Inter-residue space to Conformer
    distances = distances*deviation + mean
    pdb_from_interresidue(u_ens, CA_atoms, distances, f"{i}")
    
    # go to next point
    p[0] = p[0] + st
    i += 1
    initial_guess = distances.copy()
    
    
    # Inter-residue space to Conformer
    distances = distances*deviation + mean
    pdb_from_interresidue(u_ens, CA_atoms, distances, f"{i}")
    
    p[0] = p[0] + st
    i += 1


