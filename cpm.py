import numpy as np 
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns


def read_mats(iplist):

    x=[pd.read_csv(m,sep='\t',header=False) for m in iplist]
    x=[df.dropna(axis=1).values for df in x]
    ipmats=np.stack(x,axis=2)

    return ipmats

def run_cpm(ipmats,pheno):

    numsubs=ipmats.shape[2]
	ipmats=np.reshape(ipmats,[-1,numsubs])

	for loo in range(0,numsubs):
		
		tmats=np.delete(ipmats,[loo],axis=1)
		tpheno=np.delete(pheno,[loo],axis=0)

		

