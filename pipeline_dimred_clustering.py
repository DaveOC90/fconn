import h5py
import time
import os, sys
import glob
from functools import reduce
import pickle
import argparse

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

def generate_tsne_plots_multiple_measures():    

    # Load and reshape GSR corrected resting state leading eigenvectors
    print("Loading Data")
    rest_dct=utils.load_mat_v73('HCPDataStruct_GSR_REST_LR_LE.mat')
    rest_le=rest_dct['Leading_Eig'].value.T
    rest_le=np.reshape(rest_le,[865,1200,268])

    # Load and reshape GSR corrected working memory leading eigenvectors
    wm_dct = utils.load_mat_v73('HCPDataStruct_GSR_WM_LR_LE.mat')
    wm_le=wm_dct['Leading_Eig'].value.T
    wm_le=np.reshape(wm_le,[858,405,268])

    # Load GSR corrected working memory time series
    print("Generating static corr mats")
    ts_parcel_wm=utils.load_mat_v73('HCPDataStruct_GSR_WM_LR.mat')
    x=ts_parcel_wm['data_struct']['WM_LR']
    ## create dictionary of static correlation matrices
    wm_dct={s[0]:np.corrcoef(s[1].value.T) for s in x.items()}
    ## Record subject ids
    wmsubs=wm_dct.keys()
    ## Create one matrix with data
    wm_sc=np.stack([wm_dct[k] for k in wm_dct.keys()],axis=2)
    wm_sc=np.transpose(wm_sc,[2,0,1])
    
    # Load GSR corrected resting state time series
    ts_parcel_rest=utils.load_mat_v73('HCPDataStruct_GSR_REST_LR.mat')
    x=ts_parcel_rest['data_struct']['REST_LR']
    ## create dictionary of static correlation matrices
    rest_dct={s[0]:np.corrcoef(s[1].value.T) for s in x.items()}
    ## Record subject ids
    restsubs=rest_dct.keys()
    ## Create one matrix with data
    rest_sc=np.stack([rest_dct[k] for k in rest_dct.keys()],axis=2)
    rest_sc=np.transpose(rest_sc,[2,0,1])
    
    
    # Filter subs based on whats common to both modalities
    subs_combo=list(sorted(set(restsubs).intersection(set(wmsubs))))
    rest_mask=np.array([1 if rs in subs_combo else 0 for rs in restsubs],dtype='bool')
    wm_mask=np.array([1 if wms in subs_combo else 0 for wms in wmsubs],dtype='bool')

    # Apply to matrices
    rest_le=rest_le[rest_mask,:,:]
    wm_le=wm_le[wm_mask,:,:]
    rest_sc=rest_sc[rest_mask,:,:]
    wm_sc=wm_sc[wm_mask,:,:]
    
    # Take first fifty subjects
    rest_le=rest_le[:100,:,:]
    wm_le=wm_le[:100,:,:]
    rest_sc=rest_sc[:100,:,:]
    wm_sc=wm_sc[:100,:,:]

    # Delete some stuff to reduce memory used
    del rest_dct
    del wm_dct

    # Setup some lists to gather some stuff during processing
    data_gather=[]
    data_withevs_gather=[]
    df_gather=[]


    # Iterate over subjects
    for nsub in range(0,100):

        # Assign sub id
        subname=subs_combo[nsub].replace('sub','')
        # Find possible working memory spreadsheets
        fpaths=glob.glob(f'HCP-WM-LR-EPrime/{subname}/{subname}_3T_WM_run*_TAB_filtered.csv')

        if fpaths:

            # Theres only one file we want
            fpath=fpaths[0]
            # Load working memory event df 
            csv_opname=f'pca_dfs/dimred_events_{subname}_pcatsne.csv'

            # Calculate first EV of static mats and aggregate all LEs across one subject
            wm_sc_le=np.vstack(np.squeeze(calc_eigs(wm_sc[nsub,:,:],numevals=1)['EigVecs'])).T
            rest_sc_le=np.vstack(np.squeeze(calc_eigs(rest_sc[nsub,:,:],numevals=1)['EigVecs'])).T
            wm_rest_concat=np.concatenate([wm_le[nsub,:,:],wm_sc_le,rest_le[nsub,:,:],rest_sc_le])
            # This gathers all data, including that without event DF
            data_gather.append(wm_rest_concat)

            if not os.path.isfile(csv_opname):

                print(f"processing {subname}")

                # Caculate PCA
                pca_comps=data_reduction.return_pca_comps(wm_rest_concat.T,n_components=3)
                pca_df=pd.DataFrame(pca_comps.T,columns=['x','y','z'])

                # Setup input to TSNE
                iplist=[{'data':wm_rest_concat,'perplexity':str(i)} for i in range(10,60,20)]

                PCAinit = pca_comps.T[:,:2]/np.std(pca_comps.T[:,0])*.0001

                iplist.append({'data':wm_rest_concat,'perplexity':'30','initialization':PCAinit})

                # TSNE path to save unique runs
                tsne_dir=os.path.join('tsne_runs',subname)
                if not os.path.isdir(tsne_dir):
                    os.makedirs(tsne_dir)

                # Run fast TSNE
                tsne_dict_run=data_reduction.run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)

                # Embed key names in column headers and produce tsne df
                #tsne_dict={k:pd.DataFrame(tsne_dict_run[k].values,columns=list(map(lambda x : k+'_'+x,tsne_dict_run[k].columns))) for k in tsne_dict_run.keys()} 
                #tsne_df_merged = reduce(lambda  left,right: pd.merge(left,right,left_index=True,right_index=True,how='outer'),list(tsne_dict.values()))
                
                # Convert TSNE results to DF and reorganize columns
                tsne_df_merged=pd.concat(tsne_dict_run)
                tsne_df_merged=tsne_df_merged.reset_index().drop('level_1',axis=1)
                tsne_df_merged=tsne_df_merged.rename({'level_0':'tsneargs'},axis=1)

                num_tsne=len(tsne_dict_run)
                num_dim_red=num_tsne+1

                # Data Type Label assignment
                num_lbls=[1 for i in range(0,405)]+[2] \
                +[3 for i in range(0,1200)]+[4]
                word_lbls=['wm_le' for i in range(0,405)]+['wm_sc'] \
                +['rest_le' for i in range(0,1200)]+['rest_sc']

                # Concatenate both PCA and TSNE
                #dim_red_df=pd.merge(pca_df,tsne_df_merged,left_index=True,right_index=True,how='outer')
                dim_red_df=pd.concat([pca_df,tsne_df_merged],sort=False)

                # Apply some labels to DF
                dim_red_df['Type']=word_lbls*num_dim_red
                dim_red_df['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*num_dim_red)
                dim_red_df['subid']=np.repeat(subname,1607*num_dim_red)
                dim_red_df['method']=np.repeat(['pca']+['tsne']*num_tsne,1607)

                # Read event df
                event_df=pd.read_csv(fpath,index_col=0)

                # Merge event df and dimensionality reduction data
                dimred_event_df=pd.merge(dim_red_df,event_df,on='VolumeAssignment',how='outer')
                # Sort DF
                dimred_event_df=dimred_event_df.sort_values(['method','tsneargs','VolumeAssignment'],axis=0)
                dimred_event_df=dimred_event_df.reset_index(drop=True)
                # Write DF
                dimred_event_df.to_csv(csv_opname)


            else:
                # If DF exists skip processing and read
                print(f"{subname} already processed, loading file {csv_opname}")
                dimred_event_df=pd.read_csv(csv_opname,index_col=0)

            # Gather
            df_gather.append(dimred_event_df)

            data_withevs_gather.append(wm_rest_concat)
        else:
            print('No EVs!!!!')



    word_lbls=['wm_le' for i in range(0,405)]+['wm_sc'] \
    +['rest_le' for i in range(0,1200)]+['rest_sc']


    
    # Aggregate all the DFs
    bigdf=pd.concat(df_gather)
    #rest_wm_agg=np.concatenate(data_gather)
    bigdf['subid']=bigdf.subid.astype('str')

    # Aggregate all LEs with event data
    rest_wm_agg_wevs=np.concatenate(data_withevs_gather)

    # PCA of all
    
    pca_comps=data_reduction.return_pca_comps(rest_wm_agg_wevs.T,n_components=3)
    big_pca_df=pd.DataFrame(pca_comps.T,columns=['x','y','z'])
    x,y=big_pca_df.shape
    big_pca_df['VolumeAssignment']=np.reshape(np.repeat(np.vstack(np.arange(1,1608)),59,axis=1).T,[1607*59,1])
    big_pca_df['method']=np.repeat('pca_59subs',x)
    # Apply some labels to DF
    big_pca_df['Type']=word_lbls*59
    big_pca_df['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*59)
    big_pca_df['subid']=np.repeat(bigdf.subid.unique(),1607)




    # Setup input to TSNE
    iplist=[{'data':rest_wm_agg_wevs,'perplexity':str(i)} for i in range(10,60,20)]

    iplist=[{'data':rest_wm_agg_wevs,'perplexity':'30','initialization':pca_comps_2.T}]


    # TSNE path to save unique runs
    tsne_dir=os.path.join('tsne_runs','59subs')
    if not os.path.isdir(tsne_dir):
        os.makedirs(tsne_dir)

    # TSNE of All
    #fast_tsne_LE_33=data_reduction.run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)
    fast_tsne_LE_59=data_reduction.run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)

    num_tsne=len(fast_tsne_LE_59)

    # Create tsne DF
    big_tsne_df_merged=pd.concat(fast_tsne_LE_59)
    big_tsne_df_merged=big_tsne_df_merged.reset_index().drop('level_1',axis=1)
    big_tsne_df_merged=big_tsne_df_merged.rename({'level_0':'tsneargs'},axis=1)


    x,y=big_tsne_df_merged.shape
    # Apply some labels to DF
    big_tsne_df_merged['VolumeAssignment']=np.reshape(np.repeat(np.vstack(np.arange(1,1608)),59*num_tsne,axis=1).T,[1607*59*num_tsne,1])
    big_tsne_df_merged['method']=np.repeat('tsne_59subs',x)
    big_tsne_df_merged['Type']=word_lbls*59*num_tsne
    big_tsne_df_merged['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*59*num_tsne)
    big_tsne_df_merged['subid']=np.concatenate([np.repeat(bigdf.subid.unique(),1607)]*num_tsne)


    #print("Running HDBSCAN")
    #clus_sol=cluster.run_hdbscan(data_subset.T)


    subs=bigdf.subid.unique()
    # Merge big event df with dim red dfs
    fs=glob.glob('HCP-WM-LR-EPrime/*/*.csv')
    fs_filtered=[f for f in fs if any(str(x) in f for x in subs)]
    dfs=[pd.read_csv(ff,index_col=0) for ff in fs_filtered]
    big_ev_df=pd.concat(dfs)
    big_ev_df['subid']=np.repeat(subs,176)


    # Add Event data to dim red
    pca_ev_df=pd.merge(big_pca_df,big_ev_df,on=['VolumeAssignment','subid'],how='outer')
    tsne_ev_df=pd.merge(big_tsne_df_merged,big_ev_df,on=['VolumeAssignment','subid'],how='outer')

    # tsne_dict={k:pd.DataFrame(fast_tsne_LE_33[k].values,
    #     columns=list(map(lambda x : 'allsubs_'+k+'_'+x,fast_tsne_LE_33[k].columns))) \
    #      for k in fast_tsne_LE_33.keys()} 
    # tsne_df = reduce(lambda  left,right: pd.merge(left,right,
    #     left_index=True,right_index=True,how='outer'),list(tsne_dict.values()))

    # Merging all dfs
    #big_pca_df[['x','y','z']]=bigdf[['x','y','z']].drop_duplicates().reset_index().drop('index',axis=1)
    #tsne_df[['x','y','z']]=bigdf[['x','y','z']].drop_duplicates().reset_index().drop('index',axis=1)


    #big_pca_evdf=pd.merge(big_pca_df,bigdf,on=['x','y','z'],how='outer')

    #big_pca_tsne_evdf=pd.merge(big_pca_evdf,tsne_df,on=['x','y','z'],how='outer')

    big_pca_tsne_evdf=pd.concat([pca_ev_df,bigdf,tsne_ev_df],sort=False)
    big_pca_tsne_evdf['combo']=big_pca_tsne_evdf.method+'_'+big_pca_tsne_evdf.tsneargs.astype('str')


    #clus=run_hdbscan(rest_wm_agg_wevs)
    # Stupid work around for colors
    # cmaps=['red']*405+['blue']+['green']*1200+['orange']
    # big_pca_df['color']=np.reshape(np.repeat(np.vstack(cmaps),33,axis=1).T,[1607*33,1])

    # sizes=[5]*405+[40]+[5]*1200+[40]
    # big_pca_df['sizes']=np.reshape(np.repeat(np.vstack(sizes),33,axis=1).T,[1607*33,1])

    
    
    #for i in range(1,34):
    for sub in subs:
        # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        # cmap=big_pca_tsne_evdf[big_pca_tsne_evdf.subnum == i]['color']
        # sizes=big_pca_tsne_evdf[big_pca_tsne_evdf.subnum == i]['sizes']
        # scatter_dict={'alpha':0.4,'facecolors':cmap,'edgecolors':'face','s':sizes}
        # sns.regplot(data=big_pca_tsne_evdf[big_pca_tsne_evdf.subnum == i],x='allsubs_perplex_10_x',y='allsubs_perplex_10_y',scatter_kws=scatter_dict,ax=ax1,scatter=True,fit_reg=False)
        # sns.regplot(data=big_pca_tsne_evdf[big_pca_tsne_evdf.subnum == i],x='allsubs_perplex_30_x',y='allsubs_perplex_30_y',scatter_kws=scatter_dict,ax=ax2,scatter=True,fit_reg=False)
        # sns.regplot(data=big_pca_tsne_evdf[big_pca_tsne_evdf.subnum == i],x='allsubs_perplex_50_x',y='allsubs_perplex_50_y',scatter_kws=scatter_dict,ax=ax3,scatter=True,fit_reg=False)
        subdf=big_pca_tsne_evdf[big_pca_tsne_evdf.subid == sub]
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        #for hue in ['VolumeAssignment']:
        for hue in ['Type','BlockType','ACC']:
            print(hue)

            plt_opname='subs_66_plots/'+sub+'_dimred_'+hue+'_withpcatsne.png'

            if not os.path.isfile(plt_opname):

                plt.clf()

                sns.lmplot(data=subdf,x='x',y='y',col='combo',col_wrap=3, fit_reg=False,
                    sharex=False,sharey=False,hue=hue,scatter_kws={'alpha':0.3},
                    legend=False)
                
                plt.tight_layout()
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                #plt.show()
                plt.savefig(plt_opname)
                plt.close()
            else:
                pass

def phase_based_tsne():
    
    # Load GSR corrected working memory time series
    print("Generating static corr mats")
    if not os.path.isfile('wm_ts.npy'):
        ts_parcel_wm=utils.loadmatv73_tree('../HCPDataStruct_GSR_WM_LR.mat')
        ts_parcel_wm=ts_parcel_wm['data_struct']['WM_LR']
        np.save('wm_ts.npy',ts_parcel_wm)
    else:
        ts_parcel_wm=np.load('wm_ts.npy').item()
    wmsubs=[k.replace('sub','') for k in ts_parcel_wm.keys()]


    ## create dictionary of static correlation matrices
    wm_sc={k:np.corrcoef(ts_parcel_wm[k].T) for k in ts_parcel_wm.keys()}
    ## Create one matrix with data
    wm_sc=np.stack([wm_sc[k] for k in wm_sc.keys()],axis=2)
    wm_sc=np.transpose(wm_sc,[2,0,1])

    # Calculate phase of ts    
    wm_angles=[np.angle(sg.hilbert(ts_parcel_wm[k],axis=0)) for k in ts_parcel_wm.keys()]
    wm_angles=np.stack(wm_angles)
    wm_mean_angles=np.mean(wm_angles,axis=1)


    # Load GSR corrected resting state time series
    if not os.path.isfile('rest_ts.npy'):
        ts_parcel_rest=utils.loadmatv73_tree('../HCPDataStruct_GSR_REST_LR.mat')
        ts_parcel_rest=ts_parcel_rest['data_struct']['REST_LR']
        np.save('rest_ts.npy',ts_parcel_rest)
    else:
        ts_parcel_rest=np.load('rest_ts.npy').item()

    restsubs=[k.replace('sub','') for k in ts_parcel_rest.keys()]
    
    ## create dictionary of static correlation matrices
    rest_sc={k:np.corrcoef(ts_parcel_rest[k].T) for k in ts_parcel_rest.keys()}
    ## Create one matrix with data
    rest_sc=np.stack([rest_sc[k] for k in rest_sc.keys()],axis=2)
    rest_sc=np.transpose(rest_sc,[2,0,1])


    ## create dictionary of mean phase connectivity matrices
    rest_phasecon={k:np.mean(cosine_similarity(ts_parcel_rest[k])) for k in ts_parcel_rest.keys()}
    ## Create one matrix with data
    #rest_sc=np.stack([rest_sc[k] for k in rest_sc.keys()],axis=2)
    #rest_sc=np.transpose(rest_sc,[2,0,1])



    # Calculate phase of ts
    rest_angles=[np.angle(sg.hilbert(ts_parcel_rest[k],axis=0)) for k in ts_parcel_rest.keys()]
    rest_angles=np.stack(rest_angles)
    rest_mean_angles=np.mean(rest_angles,axis=1)


    # Event subs
    # Find possible working memory spreadsheets
    fpaths=glob.glob(f'../HCP-WM-LR-EPrime/*/*_3T_WM_run*_TAB_filtered.csv')
    eventsubs=[f.split('/')[2] for f in fpaths]
    
    # Filter subs based on whats common to both modalities
    subs_combo=list(sorted(set(restsubs).intersection(set(wmsubs)).intersection(set(eventsubs))))
    

    rest_mask=np.array([1 if rs in subs_combo else 0 for rs in restsubs],dtype='bool')
    wm_mask=np.array([1 if wms in subs_combo else 0 for wms in wmsubs],dtype='bool')

    # Mask arrays
    rest_mean_angles=rest_mean_angles[rest_mask,:]
    wm_mean_angles=wm_mean_angles[wm_mask,:]
    rest_angles=rest_angles[rest_mask,:,:]
    wm_angles=wm_angles[wm_mask,:,:]
    rest_sc=rest_sc[rest_mask,:,:]
    wm_sc=wm_sc[wm_mask,:,:]
    # Delete some stuff to reduce memory used
    del ts_parcel_rest
    del ts_parcel_wm


    # Setup some lists to gather some stuff during processing
    data_gather=[]
    data_withevs_gather=[]
    df_gather=[]


    dynamic_measure = 'angles'

    # Iterate over subjects
    for nsub in range(0,47):



        # Assign sub id
        subname=subs_combo[nsub]

        csv_opname=f'../phase_stuff/dimred_events_{subname}_pcatsne.csv'
        fpath=glob.glob(f'../HCP-WM-LR-EPrime/{subname}/{subname}_3T_WM_run*_TAB_filtered.csv')[0]

        # Create one phase array per sub
        wm_mang=np.vstack(np.squeeze(wm_mean_angles[nsub,:])).T
        rest_mang=np.vstack(np.squeeze(rest_mean_angles[nsub,:])).T
        wm_rest_concat=np.concatenate([wm_angles[nsub,:,:],wm_mang,rest_angles[nsub,:,:],rest_mang])



        if not os.path.isfile(csv_opname):
            print(f"processing {subname}")

            # Caculate PCA
            pca_comps=data_reduction.return_pca_comps(wm_rest_concat.T,n_components=3)
            pca_df=pd.DataFrame(pca_comps.T,columns=['x','y','z'])

            # Setup input to TSNE
            iplist=[{'data':wm_rest_concat,'perplexity':str(i)} for i in range(10,60,20)]

            PCAinit = pca_comps.T[:,:2]/np.std(pca_comps.T[:,0])*.0001

            iplist.append({'data':wm_rest_concat,'perplexity':'30','initialization':PCAinit})

            # TSNE path to save unique runs
            tsne_dir=os.path.join('../phase_stuff/tsne_runs',subname)
            if not os.path.isdir(tsne_dir):
                os.makedirs(tsne_dir)

            # Run fast TSNE
            tsne_dict_run=data_reduction.run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)

            # Convert TSNE results to DF and reorganize columns
            tsne_df_merged=pd.concat(tsne_dict_run)
            tsne_df_merged=tsne_df_merged.reset_index().drop('level_1',axis=1)
            tsne_df_merged=tsne_df_merged.rename({'level_0':'tsneargs'},axis=1)

            num_tsne=len(tsne_dict_run)
            num_dim_red=num_tsne+1

            # Data Type Label assignment
            num_lbls=[1 for i in range(0,405)]+[2] \
            +[3 for i in range(0,1200)]+[4]
            word_lbls=['wm_le' for i in range(0,405)]+['wm_sc'] \
            +['rest_le' for i in range(0,1200)]+['rest_sc']

            # Concatenate both PCA and TSNE
            #dim_red_df=pd.merge(pca_df,tsne_df_merged,left_index=True,right_index=True,how='outer')
            dim_red_df=pd.concat([pca_df,tsne_df_merged],sort=False)

            # Apply some labels to DF
            dim_red_df['Type']=word_lbls*num_dim_red
            dim_red_df['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*num_dim_red)
            dim_red_df['subid']=np.repeat(subname,1607*num_dim_red)
            dim_red_df['method']=np.repeat(['pca']+['tsne']*num_tsne,1607)

            # Read event df
            event_df=pd.read_csv(fpath,index_col=0)

            # Merge event df and dimensionality reduction data
            dimred_event_df=pd.merge(dim_red_df,event_df,on='VolumeAssignment',how='outer')
            # Sort DF
            dimred_event_df=dimred_event_df.sort_values(['method','tsneargs','VolumeAssignment'],axis=0)
            dimred_event_df=dimred_event_df.reset_index(drop=True)
            # Write DF
            dimred_event_df.to_csv(csv_opname)


        else:
            # If DF exists skip processing and read
            print(f"{subname} already processed, loading file {csv_opname}")
            dimred_event_df=pd.read_csv(csv_opname,index_col=0)

        # Gather
        df_gather.append(dimred_event_df)

        data_withevs_gather.append(wm_rest_concat)


    word_lbls=['wm_phase' for i in range(0,405)]+['wm_phase_mean'] \
    +['rest_phase' for i in range(0,1200)]+['rest_phase_mean']


    
    # Aggregate all the DFs
    bigdf=pd.concat(df_gather)
    numsubs=len(df_gather)
    #rest_wm_agg=np.concatenate(data_gather)
    bigdf['subid']=bigdf.subid.astype('str')

    # Aggregate all LEs with event data
    rest_wm_agg_wevs=np.concatenate(data_withevs_gather)

    # PCA of all
    
    pca_comps=data_reduction.return_pca_comps(rest_wm_agg_wevs.T,n_components=3)
    big_pca_df=pd.DataFrame(pca_comps.T,columns=['x','y','z'])
    x,y=big_pca_df.shape
    big_pca_df['VolumeAssignment']=np.reshape(np.repeat(np.vstack(np.arange(1,1608)),numsubs,axis=1).T,[1607*numsubs,1])
    big_pca_df['method']=np.repeat(f'pca_{numsubs}subs',x)
    # Apply some labels to DF
    big_pca_df['Type']=word_lbls*numsubs
    big_pca_df['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*numsubs)
    big_pca_df['subid']=np.repeat(bigdf.subid.unique(),1607)



    # Setup input to TSNE
    #iplist=[{'data':rest_wm_agg_wevs,'perplexity':str(i)} for i in range(10,60,20)]

    iplist=[{'data':rest_wm_agg_wevs,'perplexity':'30','initialization':pca_comps[:2,:].T}]


    # TSNE path to save unique runs
    tsne_dir=os.path.join('phase_stuff','tsne_runs',f'{numsubs}subs')
    if not os.path.isdir(tsne_dir):
        os.makedirs(tsne_dir)

    # TSNE of All
    #fast_tsne_LE_33=run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)
    fast_tsne_group=data_reduction.run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)

    num_tsne=len(fast_tsne_group)

    # Create tsne DF
    big_tsne_df_merged=pd.concat(fast_tsne_group)
    big_tsne_df_merged=big_tsne_df_merged.reset_index().drop('level_1',axis=1)
    big_tsne_df_merged=big_tsne_df_merged.rename({'level_0':'tsneargs'},axis=1)


    x,y=big_tsne_df_merged.shape
    # Apply some labels to DF
    big_tsne_df_merged['VolumeAssignment']=np.reshape(np.repeat(np.vstack(np.arange(1,1608)),numsubs*num_tsne,axis=1).T,[1607*numsubs*num_tsne,1])
    big_tsne_df_merged['method']=np.repeat(f'tsne_{numsubs}subs',x)
    big_tsne_df_merged['Type']=word_lbls*numsubs*num_tsne
    big_tsne_df_merged['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*numsubs*num_tsne)
    big_tsne_df_merged['subid']=np.concatenate([np.repeat(bigdf.subid.unique(),1607)]*num_tsne)


    #print("Running HDBSCAN")
    #clus_sol=cluster.run_hdbscan(data_subset.T)


    subs=bigdf.subid.unique()
    # Merge big event df with dim red dfs
    fs=glob.glob('../HCP-WM-LR-EPrime/*/*.csv')
    fs_filtered=[f for f in fs if any(str(x) in f for x in subs)]
    dfs=[pd.read_csv(ff,index_col=0) for ff in fs_filtered]
    big_ev_df=pd.concat(dfs)
    big_ev_df['subid']=np.repeat(subs,176)


    # Add Event data to dim red
    pca_ev_df=pd.merge(big_pca_df,big_ev_df,on=['VolumeAssignment','subid'],how='outer')
    tsne_ev_df=pd.merge(big_tsne_df_merged,big_ev_df,on=['VolumeAssignment','subid'],how='outer')


    big_pca_tsne_evdf=pd.concat([pca_ev_df,bigdf,tsne_ev_df],sort=False)
    big_pca_tsne_evdf['combo']=big_pca_tsne_evdf.method+'_'+big_pca_tsne_evdf.tsneargs.astype('str')
    
    
    for sub in subs:
        subdf=big_pca_tsne_evdf[big_pca_tsne_evdf.subid == sub]
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        pal=sns.color_palette(palette='RdBu')

        #for hue in ['VolumeAssignment']:
        for hue in ['Type','BlockType','ACC']:
            print(hue)

            plt_opname='../phase_stuff/subs_47_plots/'+sub+'_dimred_'+hue+'_withpcatsne.png'

            if not os.path.isfile(plt_opname):

                plt.clf()

                sns.lmplot(data=subdf,x='x',y='y',col='combo',col_wrap=3, fit_reg=False,
                    sharex=False,sharey=False,hue=hue,scatter_kws={'alpha':0.3},
                    legend=False,palette=pal)
                
                plt.tight_layout()
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                #plt.show()
                plt.savefig(plt_opname)
                plt.close()
            else:
                pass




