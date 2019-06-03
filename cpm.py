import numpy as np 
import scipy as sp
import pandas as pd
#from matplotlib import pyplot as plt
#import seaborn as sns
import glob
from scipy import stats,io,special
import random
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import corr_multi
import pickle

def read_mats(iplist):

    x=[pd.read_csv(m,sep='\t',header=None) for m in iplist]
    x=[df.dropna(axis=1).values for df in x]
    ipmats=np.stack(x,axis=2)

    return ipmats


def corr2_coeff(A,B):
	# from: https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays/30143754#30143754
    # https://stackoverflow.com/questions/45403071/optimized-computation-of-pairwise-correlations-in-python?noredirect=1&lq=1
    # Rowwise mean of input arrays & subtract from input arrays themeselves

    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def train_cpm(ipmat,pheno,pthresh=0.01):

    """
    Accepts input matrices and pheno data
    Returns model
    """

    #cc=corr2_coeff(ipmat,pheno)

    
    num_pheno=len(pheno)
    df=num_pheno-2

    Rvals=corr_multi.corr_multi_cy(pheno,ipmat.T)
    tvals=(Rvals*np.sqrt(df))/np.sqrt(1-Rvals**2)
    pvals=stats.t.sf(np.abs(tvals),df)*2

   
    # cc=[stats.pearsonr(pheno,im) for im in ipmat]
    # Rvals=np.array([c[0] for c in cc])
    # pvals=np.array([c[1] for c in cc])
    #rmat=np.reshape(Rvals,[268,268])
    #pmat=np.reshape(pvals,[268,268])
    
    posedges=(Rvals > 0) & (pvals < pthresh)
    posedges=posedges.astype(int)
    negedges=(Rvals < 0) & (pvals < pthresh)
    negedges=negedges.astype(int)
    pe=ipmat[posedges.flatten().astype(bool),:]
    ne=ipmat[negedges.flatten().astype(bool),:]
    pe=pe.sum(axis=0)/2
    ne=ne.sum(axis=0)/2




    if np.sum(pe) != 0:
        fit_pos=np.polyfit(pe,pheno,1)
    else:
        fit_pos=[]

    if np.sum(ne) != 0:
        fit_neg=np.polyfit(ne,pheno,1)
    else:
        fit_neg=[]


    return fit_pos,fit_neg,posedges,negedges


def testcorr():
    ipdata=io.loadmat('../../Fingerprinting/ipmats.mat')
    ipmats=ipdata['ipmats']
    pmatvals=ipdata['pmatvals'][0]
    ipmats_res=np.reshape(ipmats,[-1,843])
    pmats_rep=np.repeat(np.vstack(pmatvals),71824,axis=1)
    cc=corr2_coeff(ipmats_res,pmats_rep.T)

    return cc

def run_validate(ipmats,pheno,cvtype):

    numsubs=ipmats.shape[2]
    ipmats=np.reshape(ipmats,[-1,numsubs])

    cvstr_dct={
    'LOO' : numsubs,
    'splithalf' : 2,
    '5k' : 5,
    '10k' : 10}


    if type(cvtype) == str:
        if cvtype not in cvstr_dct.keys():
            raise Exception('cvtype must be LOO, 5k, 10k, or splithalf (case sensitive)')
        else:
            knum=cvstr_dct[cvtype]
    elif type(cvtype) == int:
        knum=cvtype

    else:
        raise Exception('cvtype must be an int, representing number of folds, or a string descibing CV type')



    bp,bn,ba,pe,ne=kfold_cpm(ipmats,pheno,numsubs,knum)

    bp_res=np.reshape(bp,numsubs)
    bn_res=np.reshape(bn,numsubs)
    ba_res=np.reshape(ba,numsubs)

    Rpos=stats.pearsonr(bp_res,ba_res)[0]
    Rneg=stats.pearsonr(bn_res,ba_res)[0]


    return Rpos,Rneg,pe,ne,bp_res,bn_res,ba_res
    


def kfold_cpm(ipmats,pheno,numsubs,k):
    randinds=np.arange(0,numsubs)
    random.shuffle(randinds)

    samplesize=int(np.floor(float(numsubs)/k))

    behav_pred_pos=np.zeros([k,samplesize])
    behav_pred_neg=np.zeros([k,samplesize])

    behav_actual=np.zeros([k,samplesize])

    nedges=ipmats.shape[0]

    posedge_gather=np.zeros(nedges)
    negedge_gather=np.zeros(nedges)

    for fold in range(0,k):
        print("Running fold:",fold+1)
        si=fold*samplesize
        fi=(fold+1)*samplesize


        if fold != k-1:
            testinds=randinds[si:fi]
        else:
            testinds=randinds[si:]

        traininds=randinds[~np.isin(randinds,testinds)]


        trainmats=ipmats[:,traininds]
        trainpheno=pheno[traininds]
 
        testmats=ipmats[:,testinds]
        testpheno=pheno[testinds]

        behav_actual[fold,:]=testpheno


        pos_fit,neg_fit,posedges,negedges=train_cpm(trainmats,trainpheno)

        pe=np.sum(testmats[posedges.flatten().astype(bool),:], axis=0)/2
        ne=np.sum(testmats[negedges.flatten().astype(bool),:], axis=0)/2


        posedge_gather=posedge_gather+posedges.flatten()
        negedge_gather=negedge_gather+negedges.flatten()

        if len(pos_fit) > 0:
            behav_pred_pos[fold,:]=pos_fit[0]*pe + pos_fit[1]
        else:
            behav_pred_pos[fold,:]='nan'

        if len(neg_fit) > 0:
            behav_pred_neg[fold,:]=neg_fit[0]*ne + neg_fit[1]
        else:
            behav_pred_neg[fold,:]='nan'

    posedge_gather=posedge_gather/k
    negedge_gather=negedge_gather/k


    return behav_pred_pos,behav_pred_neg,behav_actual,posedge_gather,negedge_gather


def sample_500(ipmats,pheno,cvtype):

    numsubs=ipmats.shape[2]

    randinds=np.arange(0,numsubs)
    random.shuffle(randinds)

    randinds500=randinds[:500]

    ipmats_rand=ipmats[:,:,randinds500]
    pheno_rand=pheno[randinds500]

    opdict={}

    Rpos_loo,Rneg_loo=run_validate(ipmats_rand,pheno_rand,'LOO')
    
    Rpos_2k,Rneg_2k=run_validate(ipmats_rand,pheno_rand,'splithalf')

    Rpos_5k,Rneg_5k=run_validate(ipmats_rand,pheno_rand,'5k')

    Rpos_10k,Rneg_10k=run_validate(ipmats_rand,pheno_rand,'10k')

    opdict['LOO_Rpos'] = Rpos_loo
    opdict['LOO_Rneg'] = Rneg_loo
    opdict['2k_Rpos'] = Rpos_2k
    opdict['2k_Rneg'] = Rneg_2k
    opdict['5k_Rpos'] = Rpos_5k
    opdict['5k_Rneg'] = Rneg_5k
    opdict['10k_Rpos'] = Rpos_10k
    opdict['10k_Rneg'] = Rneg_10k
    opdict['Sample_Indices']=randinds500

    return opdict







def run_cpm(args):
    niters=100
    ipmats,pmats,tp,readfile=args

    print('timepoint: ',tp)
    

    if readfile == True:
        ipmats=pickle.load(open(ipmats,'rb'))
        ipmats=np.transpose(ipmats,[1,2,0])
    
    numsubs=ipmats.shape[2]

    Rvals=np.zeros((niters,1))
    randinds=np.arange(0,numsubs)
        
    pe_gather=np.zeros(268*268)
    ne_gather=np.zeros(268*268)
    bp_gather=[]
    bn_gather=[]
    ba_gather=[]
    randinds_gather=[]

    

    for i in range(0,niters):
        print('iter: ',i)

        random.shuffle(randinds)
        randinds400=randinds[:400]
  
        ipmats_rand=ipmats[:,:,randinds400]
        pmats_rand=pmats[randinds400]


        Rp,Rn,pe,ne,bp,bn,ba=run_validate(ipmats_rand,pmats_rand,'splithalf')
        Rvals[i]=Rp
        pe_gather=pe_gather+pe
        ne_gather=ne_gather+ne
        bp_gather.append(bp)
        bn_gather.append(bn)
        ba_gather.append(ba)
        randinds_gather.append(randinds400)

    pe_gather=pe_gather/niters
    ne_gather=ne_gather/niters
    bp_gather=np.stack(bp_gather)
    bn_gather=np.stack(bn_gather)
    ba_gather=np.stack(ba_gather)
    randinds_gather=np.stack(randinds_gather)

    opdict={}
    opdict['tp']=tp
    opdict['rvals']=Rvals
    opdict['posedges']=pe_gather
    opdict['negedges']=ne_gather
    opdict['posbehav']=bp_gather
    opdict['negbehav']=bn_gather
    opdict['actbehav']=ba_gather
    opdict['randinds']=randinds_gather

    return opdict




def shred_data_run_hcp():
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
    phenofilt=pheno[pheno.Subject.isin(usesubs)]
    pmatvals=phenofilt['PMAT24_A_CR'].values

    return ipmats,pmatvals,usesubs


def shred_data_run_pnc():
    iplist=sorted(glob.glob('*matrix.txt'))
    pheno=pd.read_csv('phenotypes_info.csv')
    df_filter=pheno[['SUBJID','pmat_cr']]
    df_filter=df_filter.dropna()
    subs_mats=[i.split('_')[0] for i in iplist]
    subs_pheno=list(map(str,df_filter.SUBJID.unique()))
    subs_pheno=[sp.split('.')[0] for sp in subs_pheno]
    substouse=sorted(list(set(subs_mats) & set(subs_pheno)))

    iplist=[ip for ip in iplist if any([s in ip for s in substouse])]

    mats=[pd.read_csv(m,sep='\t',header=None).dropna(axis=1).values for m in iplist]
    ipmats=np.stack(mats,axis=2)
    pmatvals=df_filter.sort_values('SUBJID').pmat_cr.values
    

    return ipmats,pmatvals,substouse
