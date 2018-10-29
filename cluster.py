import h5py
import time
import os, sys

import numpy as np
import pandas as pd

if (os.name == 'posix' and "DISPLAY" in os.environ) or (os.name == 'nt'):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns
elif os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('agg')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns


import hdbscan
import sklearn.cluster
import scipy.cluster
from scipy import stats,io
import sklearn.datasets
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm

time_samples = [1000, 2000, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000,
               1000000, 2500000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000]

def get_timing_series(data, quadratic=True):
    '''
    Taken from: https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
    '''
    if quadratic:
        data['x_squared'] = data.x**2
        model = sm.ols('y ~ x + x_squared', data=data).fit()
        predictions = [model.params.dot([1.0, i, i**2]) for i in time_samples]
        return pd.Series(predictions, index=pd.Index(time_samples))
    else: # assume n log(n)
        data['xlogx'] = data.x * np.log(data.x)
        model = sm.ols('y ~ x + xlogx', data=data).fit()
        predictions = [model.params.dot([1.0, i, i*np.log(i)]) for i in time_samples]
        return pd.Series(predictions, index=pd.Index(time_samples))


def benchmark_algorithm(dataset_sizes, cluster_function, function_args, function_kwds,
                        dataset_dimension=10, dataset_n_clusters=10, max_time=45, sample_size=2):
    '''
    Taken from: https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
    '''
    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result = np.nan * np.ones((len(dataset_sizes), sample_size))
    for index, size in enumerate(dataset_sizes):
        for s in range(sample_size):
            print("Running for dataset size: ",size,"Features",dataset_dimension)
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            data, labels = sklearn.datasets.make_blobs(n_samples=size,
                                                       n_features=dataset_dimension,
                                                       centers=dataset_n_clusters)

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time

            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time:
                result[index, s] = time_taken
                return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                               result.flatten()]).T, columns=['x','y'])
            else:
                result[index, s] = time_taken

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                   result.flatten()]).T, columns=['x','y'])


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

def plot_3d(ipdata,labels=None,dsply=False,sve=False,savename=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(ipdata[0,:],ipdata[1,:],ipdata[2,:],s=10,c=labels)
    if dsply:
        plt.show()
    if sve and savename:
        plt.savefig(savename)
    elif sve and not savename:
        raise Exception("Must specifiy savename if sve == True")


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
    rest_le=rest_dict['Leading_Eig'].value

    wm_dict = load_mat_v73('HCPDataStruct_GSR_WM_LR_LE.mat')
    wm_le=wm_dict['Leading_Eig'].value


    rest_wm_le=np.concatenate([rest_le,wm_le],axis=1)

    data_subset=rest_wm_le[:,:120000]

    print("Running HDBSCAN")
    clus_sol=run_hdbscan(data_subset.T)

    print("Generating PCA Comps")
    pca_comps=return_pca_comps(data_subset,n_components=3)

    if os.name == 'posix' and "DISPLAY" in os.environ:
        print("Plotting....")
        plot_3d(pca_comps)

    
