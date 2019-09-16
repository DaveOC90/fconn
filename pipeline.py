import h5py
import time
import os, sys
import glob
from functools import reduce
import pickle
import argparse
import pdb
import warnings

import numpy as np
import pandas as pd
import random

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
import scipy as sp


sys.path.append('/home/dmo39/gitrepos/fconn/')
import cluster
import data_reduction
import plotting
import dfc
import utils
import cpm


from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool



if __name__ == '__main__':



    ##$ Config  $##
    

    parser=argparse.ArgumentParser(description='Run split half CV CPM on instantaneous and windowed connectivity matrices')
    parser.add_argument('timeseries',help="Path to parcellated timeseries to be used to calculate dynamic connectivity")
    parser.add_argument('prediction_target',help="Path to csv with target predictions, with rows as timepoints and columns as subjects")
    parser.add_argument('subject_list',help="Path to csv/numpy file")
    parser.add_argument('nsubs',type=int,help='Number of subs in group pool')
    parser.add_argument('nsubs_to_run',type=int,help='Number of subs to randomly sample in each CPM run')
    parser.add_argument('workdir')
    parser.add_argument('window_lengths',help='For window lengths of 1,2 and 3, pass like so: "1,2,3" ')
    parser.add_argument('window_anchor',help='Must be start middle or end')
    parser.add_argument('result_name',help='thing to append to end of resultfile')
    parser.add_argument('--trim_data',action='store_true',help='Remove time points from end of timeseries')
    parser.add_argument('--keepvols',type=int,help='Specify number of vols to keep')
    parser.add_argument('--shuffle',action='store_true')
    parser.add_argument('--calc_evec',action='store_true')
    
    args=parser.parse_args()

    ipfile=args.timeseries
    prediction_target=args.prediction_target
    subject_list=args.subject_list
    nsubs=args.nsubs
    workdir=args.workdir
    subs_to_run=args.nsubs_to_run
    trim_data=args.trim_data
    shuff=args.shuffle
    calc_evec=args.calc_evec
    avtps_list=list(map(int,args.window_lengths.split(',')))
    window_anchor=args.window_anchor
    result_name=args.result_name

   
    

    #ipfile='wm_ts.npy'
    #prediction_target='/data15/mri_group/dave_data/dcpm/RT_inputtoCPM.csv'
    #subject_list='substouse.npy'
    #nsubs=10
    #subs_to_run=10
    #workdir='/data15/mri_group/dave_data/dcpm/rawDfcPlusModel_gsr_400_400_continuousperf/'
    #trim_data=False
    #shuff=False
    #calc_evec=False
    #avtps_list=[1,5,10,20,30,60,120,240,300,350,405]
    #avtps_list=[21]
    #window_anchor='middle'

    ##$ Config  $##

    ### Establishing paths and options ###
    if (trim_data) and not (type(args.keepvols) == int):
        raise Exception('If you want trimvols, must also set number of vols to keep')
    else:
        keepvols=args.keepvols

    if not os.path.isdir(workdir):
        raise Exception('Working directory specified does not exist')

    insta_pc_dir=os.path.join(workdir,'insta_pc')
    resdir=os.path.join(workdir,'results')

    for fpath in [insta_pc_dir,resdir]:
        if not os.path.isdir(fpath):
            os.makedirs(fpath)


    ### Reading in data, subids and timepoints ###

    print("Gathering subject IDs")

    if not os.path.isfile(subject_list):
        subject_list=utils.produce_sublist(subject_list)
    else:
        subject_list=list(np.load(subject_list))


    subject_list=subject_list[:nsubs]

    

    print("Loading data")

    ts_parcel, subs = utils.load_timeseries('',ipfile,'','')

    
    
    if not all([s in subs for s in subject_list]):
        missing=[s for s in subject_list if s not in subs]
        raise Exception('The following subjects do not have imaging data: '+','.join(missing))




    prediction_target=pd.read_csv(prediction_target,index_col=0)
    pt_subs=prediction_target.columns




    if not all([s in pt_subs for s in subject_list]):
        missing=[s for s in pt_subs if s not in subs]
        raise Exception('The following subjects do not have prediction targets: '+','.join(missing))


    # Reduce prediction_target to pertinent subjects
    prediction_target=prediction_target[subject_list]
    # Pick out timepoints with no nan values
    tps_with_RT=prediction_target.dropna().index.values.astype(int)
    
    


    ## Attempting to have varying subjectlist by timepoint
    # Pick timepoints with no response
    mask=(prediction_target != 0) & ~prediction_target.isna()
    # Apply mask to subject list to create array of varying sublists
    sublistarr=np.array(subject_list)
    pred_target_subsbytp=mask.apply(lambda x : sublistarr[x],axis=1).values
    # Keep subs that are excluded by timepoint
    subbytp_exclude=mask.apply(lambda x : sublistarr[~x],axis=1).values
    
    # Turn prediction target into array
    prediction_target=prediction_target.values


    
    ### Enacting options if specified ###

    if trim_data:
        print("Trimming timeseries selected, trimming to ",keepvols," volumes")
        ts_parcel={k:ts_parcel[k][:keepvols,:] for k in ts_parcel}

    
    if shuff:
        print("Shuffling timeseries selected")
        ## WM Shuffle
        randinds=[np.random.permutation(405) for i in range(0,len(ts_parcel))]
        ts_parcel={k:ts_parcel[k][randinds[i],:] for i,k in enumerate(ts_parcel)}

        randdict={}
        randdict['Randinds']=randinds
        randdict['keys']=list(ts_parcel.keys())
        randdict['filteredsubs']=subject_list
        savepath=os.path.join(resdir,'randinds_subs.npy')
        np.save(savepath,randdict)




    
    ### Iterating over window lengths to calculate dFC ###
    ### and run CPM                                    ###    

    print('Running analysis')
    for avtps in avtps_list:


        # Figuring out start and end timepoints based on imaging data and 
        # prediction targets available
        beginshift_dct={
        'start':0,
        'middle':np.ceil(avtps/2).astype(int)-1,
        'end':avtps-1}

        endshift_dct={
        'start':avtps-1,
        'middle':np.ceil(avtps/2).astype(int)-1,
        'end':0}

        beginshift=beginshift_dct[window_anchor]
        endshift=endshift_dct[window_anchor]

        # Put in catch so code doesnt try to include tps not in imaging data
        tps_to_run_image=set(range(0+beginshift,405-endshift))
        tps_to_run=sorted(list(set.intersection(set(tps_with_RT),tps_to_run_image)))





        # Determine output names of phase connectivity data
        opnames_gather_meanphase=[os.path.join(insta_pc_dir,'pc_tp_'+str(tp).zfill(3)+'_av'+str(avtps).zfill(3)+'_'+window_anchor+'.pkl') for tp in tps_to_run]



        # Figure out if phase connectivity also exists
        ogm_mask=np.array([os.path.isfile(ogm) for ogm in opnames_gather_meanphase])

        if not all(ogm_mask):
           
            print('Calculating instantaneous PC \n Starting threading avtps: ',avtps)
            tps_to_run_arr=np.array(tps_to_run)
            tps_to_dump=list(tps_to_run_arr[~ogm_mask])


            
            thread_ips_pcmats=[(ts_parcel['sub'+subject_list[j]],insta_pc_dir,avtps,window_anchor,str(j).zfill(3),tps_to_dump) for j in range(0,len(subject_list))] # formerly nsubs

            with ThreadPool(15) as p:
                x=p.map(dfc.meanphase_dump,thread_ips_pcmats)


            p.join()

            aggfiles=np.stack(x).T

            tps_in_aggfiles=list(map(lambda x: x.split('_')[x.split('_').index('tp')+1],aggfiles[:,0]))
            tps_in_aggfiles=np.array(tps_in_aggfiles).astype(int)
            include_arr=np.isin(tps_in_aggfiles,tps_to_run)
            aggfiles_to_delete=aggfiles[~include_arr,:]
            aggfiles_to_include=aggfiles[include_arr,:]

            for aggdel in aggfiles_to_delete.flatten():
                 os.remove(aggdel)

            ogm_newlist=list(np.array(opnames_gather_meanphase)[~ogm_mask])
            ipfiles=list(zip(aggfiles_to_include,ogm_newlist))
     
            print("Gathering instantaneous PC into timelocked matrix across subs")

            with ThreadPool(15) as p:
                cpmfiles=p.map(dfc.gather_meanphase,ipfiles)
    
            p.join()

            cpmfiles=opnames_gather_meanphase

        else:
            cpmfiles=opnames_gather_meanphase
            print("input matrices already exist")



        # Pair down prediction target
        #pmats=pmats[:nsubs]
        #prediction_target=prediction_target[:,:nsubs]




        if calc_evec:
            raise Exception('Not sure things are indexing adequately')
            evec_ipfiles=[(cpmfiles[fnum],cpmfiles[fnum].replace('.pkl','_evecs.pkl')) for fnum in tps_to_run]
            with Pool(15) as p:
                evecfiles=p.map(dfc.calcLeadingEig_dump,evec_ipfiles)

            cpm_ipfiles=[(evecfiles[fnum],prediction_target[fnum,:],fnum,True,subs_to_run) for fnum in tps_to_run]
            cpm_files=[cpmfiles[fnum].replace('.pkl','_evecs.pkl') for fnum in tps_to_run]

        else:

            cpm_ipfiles=[(cpmfiles[i],prediction_target[fnum,:],fnum,True,subs_to_run,mask.values[fnum,:],subject_list) for i,fnum in enumerate(tps_to_run)]

        print("Running CPM")


        
        
        with Pool(15) as p:
            Rval_dict=p.map(cpm.run_cpm,cpm_ipfiles)

        #for cf in cpm_files:
        #    os.remove(cf)

        
        opfile_name='dCPM_tps_'+str(avtps).zfill(3)+'_'+window_anchor+'_results_'+result_name+'.npy'
        oppath=os.path.join(resdir,opfile_name)
        print("Saving results: ",oppath)
        np.save(oppath,Rval_dict)



