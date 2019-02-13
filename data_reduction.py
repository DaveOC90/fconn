import h5py
import time
import os, sys
import glob
from functools import reduce

import numpy as np
import pandas as pd



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

from fast_tsne import fast_tsne
from tsne import tsne
from leida import calc_eigs



#### Data Reduction ####
def return_pca_comps(ipdata,n_components=2):
    pca=PCA(n_components=n_components)
    pfit=pca.fit(ipdata)
    return pfit.components_



def run_fast_tsne(ipdata,perplex_list):

    opdct={}
    for pl in perplex_list:
        tsne_sol = fast_tsne(ipdata, perplexity=pl, seed=42)
        opdct['perplex_'+str(pl)]=pd.DataFrame(tsne_sol,columns=['x','y'])
    
    return opdct

def run_multiple_fast_tsne(arg_list_of_dicts,write_res=False,write_dir=''):

    if write_res and not os.path.isdir(write_dir):
        raise Exception("If write_res == True write_dir must equate to a directory")


    opdct={}

    for entry in arg_list_of_dicts:
        argsname=','.join([k+f'=entry["{k}"]' if not isinstance(entry[k],str) else k+'='+entry[k] for k in entry.keys()][1:])
        #argsname=','.join([k+f'=entry[{k}]' if not isinstance(entry[k],str) else k+'='+entry[k] for k in entry.keys()][1:])
        

        if write_res:
            arg_file_name=os.path.join(write_dir,argsname.replace(',','-').replace('"','')+'.csv')
            keyname=argsname.replace(',','-').replace('"','')


            if os.path.isfile(arg_file_name):
                opdct[keyname]=pd.read_csv(arg_file_name)
            else:

                tsne_sol=eval('fast_tsne(entry["data"],'+argsname+')')
                tsne_df=pd.DataFrame(tsne_sol,columns=['x','y'])
                tsne_df.to_csv(arg_file_name,index=False)
                opdct[keyname]=tsne_df
        else:
            tsne_sol=eval('fast_tsne(entry["data"],'+argsname+')')
            opdct[keyname]=pd.DataFrame(tsne_sol,columns=['x','y'])
            
    return opdct
