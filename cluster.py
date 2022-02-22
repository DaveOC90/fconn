import h5py
import time
import os, sys
import glob
from functools import reduce

import numpy as np
import pandas as pd

import matplotlib
if (os.name == 'posix' and "DISPLAY" in os.environ) or (os.name == 'nt'):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib.animation as animation

elif os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('agg')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib.animation as animation


import hdbscan
import sklearn.cluster
import scipy.cluster
from scipy import stats,io
import sklearn.datasets
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm
from scipy.spatial import ConvexHull

#if os.name == 'posix':
if os.path.isdir('/mnt/c/Users/david/'):
    sys.path.append('/mnt/c/Users/david/Documents/Research/gitrepos/FIt-SNE')
    sys.path.append('/mnt/c/Users/david/Documents/Research/fconn/')
#elif os.name == 'nt':
elif os.path.isdir('C:/Users/david/'):
    sys.path.append('C:/Users/david/Documents/Research/gitrepos/FIt-SNE')
    sys.path.append('C:/Users/david/Documents/Research/fconn/')
elif os.path.isdir('/home/dmo39/'):
    sys.path.append('/home/dmo39/gitrepos/FIt-SNE-master/')
    sys.path.append('/home/dmo39/gitrepos/fconn/')

#from fast_tsne import fast_tsne
#from tsne import tsne
from leida import calc_eigs



#### Clustering ####


def run_hdbscan(ipdata,eps=None):
    if eps:
        clusterer=hdbscan.HDBSCAN(algorithm='boruvka_kdtree',eps=eps)
    else:
        clusterer=hdbscan.HDBSCAN(algorithm='boruvka_kdtree')
    cf=clusterer.fit(ipdata)
    return cf




