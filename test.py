import numpy as np 
import pandas as pd 
from scipy import io
import cython
from distutils.core import setup
from Cython.Build import cythonize
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import corr_multi
import time
import timeit

#data load here
ipmats=np.load('../10simplerules_data_figs/data/ipmats.npy')
ipmats=np.reshape(ipmats,[268**2,843])
pmatfile=io.loadmat('../10simplerules_data_figs/data/hcppmats.mat') 
pmat_hcp=pmatfile['pmats_hcp']



setup(ext_modules=cythonize('./corr_multi.pyx',annotate=True))

#start=time.time()
t=timeit.Timer("corr_multi.corr_multi_cy(np.squeeze(pmat_hcp),ipmats.T)", "import corr_multi;import numpy as np;from __main__ import pmat_hcp, ipmats")
time_list=t.repeat(repeat=10,number=1)
print(np.mean(time_list),np.std(time_list))


#x=corr_multi.corr_multi_cy(np.squeeze(pmat_hcp),ipmats.T)
#print(time.time()-start)