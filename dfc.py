import h5py
import time
import os, sys
import glob
from functools import reduce

import numpy as np
import pandas as pd
import pickle

from leida import calc_eigs

import hdbscan
import sklearn.cluster
import scipy.cluster
from scipy import stats,io
from scipy import signal as sg
import sklearn.datasets
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm
from scipy.spatial import ConvexHull

#### Dynamic Estimates ####

def return_sliding_windows(ipdata,window_size=30):
    '''
    Takes a matrix of size subs x ntimepoints x rois 
    and returns a matrix of sliding window correlations
    of size subs x ntimepoints x rois x rois - window_size
    '''
    nsubs,ntps,nrois=ipdata.shape

    opdata=np.zeros([nsubs,(ntps-window_size),nrois,nrois])

    for sub in range(0,nsubs):
        for nwindow in range(0,(ntps-window_size)):
            opdata[sub,nwindow,:,:]=np.corrcoef(ipdata[sub,nwindow:(nwindow+window_size),:].T)


    return opdata


def cosine_similarity(timeseries):
    """
    Function to calculate similarity between timeseries as a
    function of the angle of the complex representation
    Takes NxM matrix, where M = number of timeseries, and 
    N = number of timepoints
    Returns a matrix of size N x M x M
    """
    n_ts=timeseries.shape[1]
    n_tp=timeseries.shape[0]
    hilt = sg.hilbert(timeseries,axis=0)
    angles = np.angle(hilt)

    pw_diff=np.array([angles[v,:] - a for v in range(0,n_tp) for a in angles[v,:]])
    pw_diff=np.reshape(pw_diff,[n_tp,n_ts,n_ts])

    cos_sim=np.cos(pw_diff)

    return cos_sim

def harmonize_evecs(evec):
    # % Make sure the largest component is negative
    # % This step is important because the same eigenvector can
    # % be returned either as V or its symmetric -V and we need
    # % to make sure it is always the same (so we choose always
    # % the most negative one)
    if np.mean(evec>0)>.5:
        evec=-evec;
    elif np.mean(evec>0)==.5 and np.sum(evec[evec>0])>-np.sum(evec[evec<0]):
        evec=-evec;

    return evec


def calc_eigs(matrices,numevals="All"):
    """
    Takes NxMxM matrix and returns eigenvalues and eigenvectors
    """
    
    if len(matrices.shape) == 3:
        nvols,nrois,_=matrices.shape
    elif len(matrices.shape) == 2:
        #print('2D, Assuming this is an ROIxROI matrix')
        nrois,_=matrices.shape
        nvols=1
    else:
        raise(Exception("Not sure about this matrix shape"))

    evals=np.zeros([nvols,nrois])
    evecs=np.zeros([nvols,nrois,nrois])
    evars=np.zeros([nvols,nrois])

    for volnum in range(0,nvols):
        #print(volnum)

        if len(matrices.shape) == 3:
            eigs=sp.linalg.eigh(matrices[volnum,:,:])
        else:
            eigs=sp.linalg.eigh(matrices)


        tevals=eigs[0]
        tevecs=eigs[1]

        tevecs=np.array([harmonize_evecs(tevecs[:,i]) for i in range(0,tevecs.shape[1])]).T

        evsort=np.argsort(tevals)
        tevals=tevals[evsort[::-1]]
        evals[volnum,:]=tevals
        evecs[volnum,:,:]=tevecs[:,evsort[-1::]]
        evars[volnum,:]=np.array([tevals[i]/np.sum(tevals,axis=None) for i in range(0,tevals.shape[0])])



        #evecs=np.array([evecs[i,:,evsort[i,:]] for i in range(0,len(evsort))])
        #evars=np.array([evals[i,:]/np.sum(evals[i,:],axis=None) for i in range(0,evals.shape[0])])


    opdict={}

    if numevals == 'All':
        opdict['EigVals']=evals
        opdict['EigVecs']=evecs
        opdict['EigVars']=evars

    else:
        opdict['EigVals']=evals[:,0:numevals]
        opdict['EigVecs']=evecs[:,:,0:numevals]
        opdict['EigVars']=evars[:,0:numevals]

    return opdict


def tsfilt(timeseries):
    """
    Demean and detrend input timeseries of one subject
    accepts array of size NxM, N = timepoints, M = timeseries
    """
    ts_detrend=sg.detrend(timeseries,axis=0)
    ts_demean=ts_detrend-ts_detrend.mean(axis=0)
    
    return ts_demean


def dotnorm(v1,v2):
    return  np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)   

def indv_leida_mats(onesubdata,numeigs=1):
    """
    """

    filtered_data=tsfilt(onesubdata)

    cos_sim_data=cosine_similarity(filtered_data)

    opdict=calc_eigs(cos_sim_data,numevals=numeigs)

    evecs=opdict['EigVecs']
    tp,ts,numevec=evecs.shape
    fcd_list=[]
    for fcdi in range(0,numevec):
        evec=np.squeeze(evecs[:,:,fcdi])
        dns=np.array([dotnorm(e1, e2) for e1 in evec for e2 in evec])
        fcd_list.append(np.reshape(dns,[tp,tp]))


    opdict['FCD']=fcd_list

    return opdict


def meanphase_dump(args):

    ts_parcel,opdir,windowtps,subid=args

    ntps=ts_parcel.shape[0]
    phasecon=cosine_similarity(ts_parcel)
    fpaths=[]


    for tp in range(0,ntps-windowtps+1):
        opfname='indvphasecon_tp_'+str(tp).zfill(3)+'_sub_'+subid+'.pkl'
        opfpath=os.path.join(opdir,opfname)
        fpaths.append(opfpath)
        mean_window_phasecon=np.mean(phasecon[tp:tp+windowtps,:,:],axis=0)
        tp_phasecon=np.expand_dims(mean_window_phasecon,0)
        pickle.dump(tp_phasecon,open(opfpath,'wb'))

    print('Final data written to: ',opfpath)

    return fpaths

def gather_meanphase(args):
    ippaths,oppath=args
    numfs=len(ippaths)

    if numfs == 1:
        fpath=ippaths[0]
        av_pc=pickle.load(open(fpath,'rb'))
        os.remove(fpath)


    else:
        gather_mats=[pickle.load(open(ipf,'rb')) for ipf in ippaths]
        av_pc=np.stack(gather_mats).squeeze()
        for ipf in ippaths:
            os.remove(ipf)

    pickle.dump(av_pc,open(oppath,'wb'))

    print('Wrote to:', oppath)

    return oppath

