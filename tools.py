import utils
import scipy
import random
import matplotlib
import tools as t
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx 
import matplotlib.pyplot as plt

from scipy import spatial
from numpy import linalg as LA
from collections import Counter
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel
from networkx.algorithms.cuts import conductance
from scipy.sparse.csgraph import connected_components
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import SpectralClustering, AffinityPropagation

def k_cluster_test(points, 
                   k, 
                   l, 
                   n_verts):
    
    M_d = spatial.distance_matrix(points,
                                  points,
                                  p=2)**2
    dist_thrshld = np.median(M_d.flatten())

    mask = M_d<dist_thrshld
    remove_diag = np.eye(M_d.shape[0])==0
    mask = remove_diag*mask

    stay_prob = np.eye(M_d.shape[0])*0.5
    d = mask.sum(0).max()
    move_prob = 1/(2*d)

    M = move_prob*mask + stay_prob
    add_self_loop = np.diag(1-M.sum(0))
    M = M + add_self_loop
    M = M.astype('float32')
    

    # create a mask so that we dont sample verteces that are not connected to anything 
    singles_mask = mask.sum(0)>0
    singles_mask = singles_mask.reshape(-1,1)

    #total number of vertices that are connected (not to itself)
    n = (mask.sum(0) > 0).sum() 

    S = np.random.random_sample((M.shape[1], 
                                 n_verts))
    S = S*singles_mask
    S = (S.max(axis=0,keepdims=1) == S)*1
    S = S.astype('float32')
    
    M_l = np.linalg.matrix_power(M, l)
    S_l = np.matmul(M_l, S)
    p_l2 = np.linalg.norm(S_l, ord=2, axis=0)**2
    sigma = 192*n_verts*k/n
    keep_idx = p_l2<sigma

    '''
    TODO: put a function that sample more vertices if it doesnt pass the sigma test'
    【・ヘ・】
    '''
    assert len(keep_idx)==n_verts, 'Sample more vertices, didnt pass sigma test'
    
    H = spatial.distance_matrix(np.swapaxes(S_l,0,1),
                                np.swapaxes(S_l,0,1),
                                p=2)**2
    remove_diag = np.eye(n_verts)*9999
    H = H+remove_diag
    H = H<=1/(4*n)

    graph = csr_matrix(H)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    print('{} islands'.format(n_components))
    if n_components>k:
        print('k={} Need more clustering （・∩・)'.format(k))
        return False
    else:
        print('k={} Good amount of clustering (･o･)'.format(k))    
        return True
    
    
def eigenDecomposition(A, plot = True, topK = 5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
    # eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eigh(L)
    eigenvalues = np.sort(np.real(eigenvalues))
    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors