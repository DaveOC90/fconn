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
import pdb

def read_mats(iplist):

    x=[pd.read_csv(m,sep='\t',header=None) for m in iplist]
    x=[df.dropna(axis=1).values for df in x]
    ipmats=np.stack(x,axis=2)

    return ipmats




def train_cpm(ipmat,pheno,pthresh=0.01, corrtype = 'pearsonr', confound=False):

    """
    Accepts input matrices (NRois x Nsubs) and pheno data
    Returns model
    """
    
    num_pheno=len(pheno)
    

    if corrtype == 'pearsonr':
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

    elif corrtype == 'partial':
        if type(confound) != np.ndarray:
             raise Exception('if corrtype is partial confounds must be specified')

        df=num_pheno-3
       
        y=pheno
        y=(y-np.mean(y))/np.std(y)

        z=confound
        z=(z-np.mean(z))/np.std(z)

        

        ipmatMean=np.vstack(np.mean(ipmat,axis=1))
        ipmatStdv=np.vstack(np.std(ipmat,axis=1))

        

        ipmatNorm=(ipmat-ipmatMean)/ipmatStdv

        Rvals=corr_multi.partial_corr(ipmatNorm.T,y,z)

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

    else:
        raise Exception('corrtype must be "pearsonr" or "partial"')



    if np.sum(pe) != 0:
        fit_pos=np.polyfit(pe,pheno,1)
    else:
        fit_pos=[]

    if np.sum(ne) != 0:
        fit_neg=np.polyfit(ne,pheno,1)
    else:
        fit_neg=[]


    return fit_pos,fit_neg,posedges,negedges


    


def kfold_cpm(ipmats,pheno,numsubs,k,corrtype,confound):
    randinds=np.arange(0,numsubs)
    random.shuffle(randinds)

    samplesize=int(np.floor(float(numsubs)/k))

    behav_pred_pos=np.zeros([k,samplesize])
    behav_pred_neg=np.zeros([k,samplesize])

    behav_actual=np.zeros([k,samplesize])

    nedges=ipmats.shape[0]

    posedge_gather=np.zeros(nedges)
    negedge_gather=np.zeros(nedges)
    pf_gather=[]
    nf_gather=[]

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

        if corrtype == 'partial':
            trainconf=confound[traininds]
            testconf=confound[testinds]
        else:
            trainconf=confound
            testconf=confound

        behav_actual[fold,:]=testpheno


        pos_fit,neg_fit,posedges,negedges=train_cpm(trainmats,trainpheno,corrtype = corrtype,confound = trainconf)

        pe=np.sum(testmats[posedges.flatten().astype(bool),:], axis=0)/2
        ne=np.sum(testmats[negedges.flatten().astype(bool),:], axis=0)/2


        posedge_gather=posedge_gather+posedges.flatten()
        negedge_gather=negedge_gather+negedges.flatten()
        pf_gather.append(pos_fit)
        nf_gather.append(neg_fit)

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


    return behav_pred_pos,behav_pred_neg,behav_actual,posedge_gather,negedge_gather, pf_gather, nf_gather



def run_validate(ipmats,pheno,cvtype,corrtype,confound):

    numsubs=ipmats.shape[1]
    #ipmats=np.reshape(ipmats,[-1,numsubs])

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



    bp,bn,ba,pe,ne,pf,nf=kfold_cpm(ipmats,pheno,numsubs,knum,corrtype,confound)

    bp_res=np.reshape(bp,numsubs)
    bn_res=np.reshape(bn,numsubs)
    ba_res=np.reshape(ba,numsubs)

    Rpos=stats.pearsonr(bp_res,ba_res)[0]
    Rneg=stats.pearsonr(bn_res,ba_res)[0]


    return Rpos,Rneg,pe,ne,bp_res,bn_res,ba_res,pf,nf




def run_cpm(args):

    '''
    Interface for multiprocessing run
    '''


    niters=50
    ipmats,pmats,tp,readfile,subs_to_run,tpmask,sublist,corrtype,confound=args


    print('timepoint: ',tp)
    

    if readfile == True:
        ipmats=pickle.load(open(ipmats,'rb'))
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')
            

    if type(tpmask) == np.ndarray and tpmask.dtype == bool:

        ipmats=ipmats[:,tpmask]
        pmats=pmats[tpmask]
        if corrtype == 'partial':
            condfound=confound[tpmask]

        if ipmats.shape[1] < subs_to_run:
            raise Exception('Not enough subs')
        if pmats.shape[0] < subs_to_run:
            raise Exception('Not enough dependent variables')

        ipmats=ipmats[:,:subs_to_run]
        pmats=pmats[:subs_to_run]
        if corrtype == 'partial':
            confound=confound[:subs_to_run]

        numsubs=subs_to_run

    elif type(tpmask) == bool and tpmask == False:

        pass
    else:
        raise Exception('Datatype of mask not recognized, must be a boolean ndarray or boolean of value "False"')
        



    #ipmats=np.arctanh(ipmats)
    #ipmats[ipmats == np.inf] = np.arctanh(0.999999)

    Rvals=np.zeros((niters,1))
    randinds=np.arange(0,numsubs)


    pe_gather=np.zeros(nrois)
    ne_gather=np.zeros(nrois)
    bp_gather=[]
    bn_gather=[]
    ba_gather=[]
    randinds_gather=[]
    pf_gather=[]
    nf_gather=[]
    pe_gather_save=[]


    for i in range(0,niters):
        print('iter: ',i)

        random.shuffle(randinds)
        randinds_torun=randinds[:subs_to_run]
        #randinds_to_run=randinds  

        ipmats_rand=ipmats[:,randinds_torun]
        pmats_rand=pmats[randinds_torun]


        Rp,Rn,pe,ne,bp,bn,ba,pf,nf=run_validate(ipmats_rand,pmats_rand,'splithalf',corrtype,confound)
        Rvals[i]=Rp
        if i < 5:
            pe_gather_save.append(pe)
        pe_gather=pe_gather+pe
        ne_gather=ne_gather+ne
        bp_gather.append(bp)
        bn_gather.append(bn)
        ba_gather.append(ba)
        randinds_gather.append(randinds_torun)
        pf_gather.append(pf)
        nf_gather.append(nf)

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
    opdict['posedgesIndv']=pe_gather_save
    opdict['negedges']=ne_gather
    opdict['posbehav']=bp_gather
    opdict['negbehav']=bn_gather
    opdict['actbehav']=ba_gather
    opdict['randinds']=randinds_gather
    opdict['posfits']=pf_gather
    opdict['negfits']=pf_gather
    opdict['sublist']=sublist

    if type(tpmask) == np.ndarray and tpmask.dtype == bool:
        opdict['tpmask']=tpmask

    return opdict


def apply_cpm(ipmats,pmats,edges,model,tpmask,readfile,subs_to_run):

    """
    Accepts input matrices, edges and model
    Returns predicted behavior
    """    

    if readfile == True:
        ipmats=pickle.load(open(ipmats,'rb'))
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')

    else:
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')

            

    if type(tpmask) == np.ndarray and tpmask.dtype == bool:

        ipmats=ipmats[:,tpmask]
        pmats=pmats[tpmask]

        if ipmats.shape[1] < subs_to_run:
            raise Exception('Not enough subs')
        if pmats.shape[0] < subs_to_run:
            raise Exception('Not enough dependent variables')

        ipmats=ipmats[:,:subs_to_run]
        pmats=pmats[:subs_to_run]

        numsubs=subs_to_run

    elif type(tpmask) == bool and tpmask == False:
        pass
    else:
        raise Exception('Datatype of mask not recognized, must be a boolean ndarray or boolean of value "False"')
        


    
    edgesum=np.sum(ipmats[edges.flatten().astype(bool),:], axis=0)/2   
    behav_pred=model[0]*edgesum + model[1]



    predscore=np.corrcoef(behav_pred,pmats)[0,1]


    return predscore

def bootstrapApplyCPM(ipmats,pmats,edges,model,tpmask,readfile,subs_to_run,niters):
    """
    Accepts input matrices, edges and model
    Bootstraps smaller samples from input matrices
    Returns array of predicted behavior
    """

    ipmats=pickle.load(open(ipmats,'rb'))

    numsubs=ipmats.shape[0]

    Rvals=np.zeros((niters,1))
    randinds=np.arange(0,numsubs)

    for i in range(0,niters):
        print('iter: ',i)

        random.shuffle(randinds)
        randinds_torun=randinds[:subs_to_run]

        ipmats_rand=ipmats[randinds_torun,:,:]
        pmats_rand=pmats[randinds_torun]

        Rvals[i,:]=apply_cpm(ipmats_rand,pmats_rand,edges,model,False,False,False)

    return Rvals



###### Old stuff ###################################

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



def testcorr():
    ipdata=io.loadmat('../../Fingerprinting/ipmats.mat')
    ipmats=ipdata['ipmats']
    pmatvals=ipdata['pmatvals'][0]
    ipmats_res=np.reshape(ipmats,[-1,843])
    pmats_rep=np.repeat(np.vstack(pmatvals),71824,axis=1)
    cc=corr2_coeff(ipmats_res,pmats_rep.T)

    return cc




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
