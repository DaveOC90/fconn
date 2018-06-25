import numpy as np 
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns


def read_mats(iplist):

    x=[pd.read_csv(m,sep='\t',header=None) for m in iplist]
    x=[df.dropna(axis=1).values for df in x]
    ipmats=np.stack(x,axis=2)

    return ipmats



def train_cpm(ipmat,pheno):

    """
    Accepts input matrices and pheno data
    Returns model
    """

    cc=[stats.pearsonr(pheno,im) for im in ipmat]
    rmat=np.array([c[1] for c in cc])
    pmat=np.array([c[0] for c in cc])
    rmat=np.reshape(rmat,[268,268])
    pmat=np.reshape(pmat,[268,268])
    posedges=(rmat > 0) & (pmat < 0.01)
    posedges=posedges.astype(int)
    negedges=(rmat < 0) & (pmat < 0.01)
    negedges=negedges.astype(int)
    pe=ipmats[posedges.flatten().astype(bool),:]
    ne=ipmats[negedges.flatten().astype(bool),:]
    pe=pe.sum(axis=0)/2
    ne=ne.sum(axis=0)/2

    if np.sum(pe) > 0:
        fit_pos=np.polyfit(pe,pmatvals,1)
    else:
       fit_pos=[]

    if np.sum(ne) > 0:
        fit_neg=np.polyfit(ne,pmatvals,1)
    else:
       fit_neg=[]

    return fit_pos,fit_neg,posedges,negedges



def run_validate(ipmats,pheno,cvtype):

    numsubs=ipmats.shape[2]
    ipmats=np.reshape(ipmats,[-1,numsubs])

    

    if cvtype == 'LOO':
        behav_pred_pos=np.zeros[numsubs]
        behav_pred_neg=np.zeros[numsubs]
        for loo in range(0,numsubs):
      
          train_mats=np.delete(ipmats,[loo],axis=1)
          train_pheno=np.delete(pheno,[loo],axis=0)
            
            test_mat=ipmats[:,loo]
            test_phenp=pheno[loo]

          pos_fit,neg_fit,posedges,negedges=train_cpm(train_mats,train_pheno)

            pe=np.sum(test_mat[posedges.flatten().astype(bool),:],axis=1)/2
            ne=np.sum(test_mat[negedges.flatten().astype(bool),:],axis=1)/2

            if pos_fit:
                behav_pred_pos[loo]=pos_fit[0]*pe + pos_fit[1]
            else:
                behav_pred_pos[loo]='nan'

            if neg_fit:
               behav_pred_neg[loo]=neg_fit[0]*ne + neg_fit[1]
            else:
                behav_pred_neg[loo]='nan'

    return behav_pred_pos,behav_pred_neg


def shred_data_run():
	mats=glob.glob('*WM*LR*_GSR*.txt')
    mats=list(sorted(mats))
    pheno=pd.read_csv('unrestricted_dustin_6_21_2018_20_47_17.csv')
    subs=[m.split('_')[0] for m in mats]
    pfilesubs=set(pheno.Subject)
    subs=[int(s) for s in subs]
    subset=set(subs)
    usesubs=list(pfilesubs.intersection(subset))
    usesubs=list(map(str,usesubs))
    usesubs=sorted(usesubs)
    iplist=[m for m in mats if any([u in m for u in usesubs])]
    x=[pd.read_csv(m,sep='\t',header=None) for m in iplist]
    x=[df.dropna(axis=1).values for df in x]
    ipmats=np.stack(x,axis=2)
    numsubs=ipmats.shape[2]
    ipmats=np.reshape(ipmats,[-1,numsubs])
    phenofilt=pheno[pheno.Subject.isin(usesubs)]
    pmatvals=phenofilt['PMAT24_A_CR'].values

    return ipmats,phenofilt