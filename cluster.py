import hdbscan
from scipy import stats,io
import h5py
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def load_mat_v73(fname):
    f_load = h5py.File(fname, 'r')
    opdict={}
    for k,v in f_load.items():
        opdict[k]=v

    return opdict


def return_pca_comps(ipdata,n_components=2):
    pca=PCA(n_components=n_components)
    pfit=pca.fit(ipdata)

    return pfit.components_

def plot_3d(ipdata,labels=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(ipdata[0,:],ipdata[1,:],ipdata[2,:],s=10,c=labels)
    plt.show()

def run_hdbscan(ipdata,eps=None):

    if eps:
        clusterer=hdbscan.HDBSCAN(algorithm='boruvka_kdtree',eps=eps)
    else:
        clusterer=hdbscan.HDBSCAN(algorithm='boruvka_kdtree')
    
    cf=clusterer.fit(ipdata)

    return cf

if __name__ == '__main__':
    
    print("Loading Data")
    rest_dict=load_mat_v73('HCPDataStruct_GSR_REST_LR_LE.mat')
    rest_le=rest_dict['Leading_Eig']

    wm_dict = load_mat_v73('HCPDataStruct_GSR_WM_LR_LE.mat')
    wm_le=wm_dict['Leading_Eig']


    rest_wm_le=np.concatenate([rest_le,wm_le],axis=1)

    data_subset=rest_wm_le[:,:120000]

    print("Running HDBSCAN")
    clus_sol=run_hdbscan(data_subset.T)

    print("Generating PCA Comps")
    pca_comps=return_pca_comps(data_subset,n_components=3)

    print("Plotting....")
    plot_3d(pca_comps)

    
