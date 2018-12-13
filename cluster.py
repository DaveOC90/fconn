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

time_samples = [1000, 2000, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000,
               1000000, 2500000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000]

def get_timing_series(data, quadratic=True):
    '''
    Taken from: https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
    '''
    if quadratic:
        data['x_squared'] = data.x**2
        model = sm.ols('y ~ x + x_squared', data=data).fit()
        predictions = [model.params.dot([1.0, i, i**2]) for i in time_samples]
        return pd.Series(predictions, index=pd.Index(time_samples))
    else: # assume n log(n)
        data['xlogx'] = data.x * np.log(data.x)
        model = sm.ols('y ~ x + xlogx', data=data).fit()
        predictions = [model.params.dot([1.0, i, i*np.log(i)]) for i in time_samples]
        return pd.Series(predictions, index=pd.Index(time_samples))


def benchmark_algorithm(dataset_sizes, cluster_function, function_args, function_kwds,
                        dataset_dimension=10, dataset_n_clusters=10, max_time=45, sample_size=2):
    '''
    Taken from: https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
    '''
    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result = np.nan * np.ones((len(dataset_sizes), sample_size))
    for index, size in enumerate(dataset_sizes):
        for s in range(sample_size):
            print("Running for dataset size: ",size,"Features",dataset_dimension)
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            data, labels = sklearn.datasets.make_blobs(n_samples=size,
                                                       n_features=dataset_dimension,
                                                       centers=dataset_n_clusters)

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time

            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time:
                result[index, s] = time_taken
                return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                               result.flatten()]).T, columns=['x','y'])
            else:
                result[index, s] = time_taken

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                   result.flatten()]).T, columns=['x','y'])


def load_mat_v73(fname):
    f_load = h5py.File(fname, 'r')
    opdict={}
    for k,v in f_load.items():
        opdict[k]=v

    return opdict

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




def return_pca_comps(ipdata,n_components=2):
    pca=PCA(n_components=n_components)
    pfit=pca.fit(ipdata)
    return pfit.components_

def plot_3d(ipdata,labels=None,dsply=False,sve=False,savename=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(ipdata[0,:],ipdata[1,:],ipdata[2,:],s=10,c=labels)
    if dsply:
        plt.show()
    if sve and savename:
        plt.savefig(savename)
    elif sve and not savename:
        raise Exception("Must specifiy savename if sve == True")


def run_hdbscan(ipdata,eps=None):
    if eps:
        clusterer=hdbscan.HDBSCAN(algorithm='boruvka_kdtree',eps=eps)
    else:
        clusterer=hdbscan.HDBSCAN(algorithm='boruvka_kdtree')
    cf=clusterer.fit(ipdata)
    return cf

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
        
        args=','.join([k+'='+entry[k] for k in entry.keys() if k != 'data'])

        if write_res:
            arg_file_name=os.path.join(write_dir,args.replace(',','-')+'.csv')
            if os.path.isfile(arg_file_name):
                opdct[args]=pd.read_csv(arg_file_name)
            else:
                tsne_sol=eval('fast_tsne(entry["data"],'+args+')')
                tsne_df=pd.DataFrame(tsne_sol,columns=['x','y'])
                tsne_df.to_csv(arg_file_name,index=False)
                opdct[args]=tsne_df
        else:
            tsne_sol=eval('fast_tsne(entry["data"],'+args+')')
            opdct[args]=pd.DataFrame(tsne_sol,columns=['x','y'])
            
    return opdct

def plot_tsne(ipdct,hue_name=''):

    nplots=len(ipdct)
    nrows=int(np.ceil(nplots/4))

    titles=list(ipdct.keys())

    if nplots < 4:
        ncols=nplots
    else:
        ncols=4

    for i,title in enumerate(titles):
        plt.subplot(nrows,ncols,i+1)
        plt.title(title)
        if hue_name:
            ncolours=len(ipdct[title][hue_name].unique())
            cmap=sns.color_palette("coolwarm", n_colors=ncolours)
            sns.scatterplot(data=ipdct[title],x='x',y='y',hue=hue_name,palette=cmap,legend=False,alpha=0.7)
        else:
            npoints=ipdct[title].shape[0]
            cmap=sns.color_palette("coolwarm", n_colors=npoints)
            sns.scatterplot(data=ipdct[title],x='x',y='y',hue=np.arange(0,npoints),palette=cmap,legend=False,alpha=0.7)
            #plt.show()


def get_data(i=0):
    x,y = np.random.normal(loc=i,scale=3,size=(2, 260))
    return x,y


def run_animation(data,anisave=False,savename=''):

    def init():
        g=sns.heatmap(np.zeros((nx, ny)), vmax=1,vmin=-1,cmap=cmap)
        return g,

    def animate(frame):
        plt.clf()
        g=sns.heatmap(data[:,:,frame], vmax=1, vmin=-1,cmap=cmap)
        return g,

    nx,ny,frames=data.shape
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, repeat = False, blit=True, interval=100)


    if anisave:
        anim.save(savename, dpi=80, writer='imagemagick')
    else:
        plt.show()



if __name__ == '__main__':
    

    # Load and reshape GSR corrected resting state leading eigenvectors
    print("Loading Data")
    rest_dct=load_mat_v73('HCPDataStruct_GSR_REST_LR_LE.mat')
    rest_le=rest_dct['Leading_Eig'].value.T
    rest_le=np.reshape(rest_le,[865,1200,268])

    # Load and reshape GSR corrected working memory leading eigenvectors
    wm_dct = load_mat_v73('HCPDataStruct_GSR_WM_LR_LE.mat')
    wm_le=wm_dct['Leading_Eig'].value.T
    wm_le=np.reshape(wm_le,[858,405,268])

    # Load GSR corrected working memory time series
    print("Generating static corr mats")
    ts_parcel_wm=load_mat_v73('HCPDataStruct_GSR_WM_LR.mat')
    x=ts_parcel_wm['data_struct']['WM_LR']
    ## create dictionary of static correlation matrices
    wm_dct={s[0]:np.corrcoef(s[1].value.T) for s in x.items()}
    ## Record subject ids
    wmsubs=wm_dct.keys()
    ## Create one matrix with data
    wm_sc=np.stack([wm_dct[k] for k in wm_dct.keys()],axis=2)
    wm_sc=np.transpose(wm_sc,[2,0,1])
    
    # Load GSR corrected resting state time series
    ts_parcel_rest=load_mat_v73('HCPDataStruct_GSR_REST_LR.mat')
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
            csv_opname=f'pca_dfs/dimred_events_{subname}.csv'

            # Calculate first EV of static mats and aggregate all LEs across one subject
            wm_sc_le=np.vstack(np.squeeze(calc_eigs(wm_sc[nsub,:,:],numevals=1)['EigVecs'])).T
            rest_sc_le=np.vstack(np.squeeze(calc_eigs(rest_sc[nsub,:,:],numevals=1)['EigVecs'])).T
            wm_rest_concat=np.concatenate([wm_le[nsub,:,:],wm_sc_le,rest_le[nsub,:,:],rest_sc_le])
            # This gathers all data, including that without event DF
            data_gather.append(wm_rest_concat)

            if not os.path.isfile(csv_opname):

                print(f"processing {subname}")

                # Caculate PCA
                pca_comps=return_pca_comps(wm_rest_concat.T,n_components=3)
                pca_df=pd.DataFrame(pca_comps.T,columns=['x','y','z'])

                # Setup input to TSNE
                iplist=[{'data':wm_rest_concat,'perplexity':str(i)} for i in range(10,60,20)]

                # TSNE path to save unique runs
                tsne_dir=os.path.join('tsne_runs',subname)
                if not os.path.isdir(tsne_dir):
                    os.makedirs(tsne_dir)

                # Run fast TSNE
                tsne_dict_run=run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)

                # Embed key names in column headers and produce tsne df
                #tsne_dict={k:pd.DataFrame(tsne_dict_run[k].values,columns=list(map(lambda x : k+'_'+x,tsne_dict_run[k].columns))) for k in tsne_dict_run.keys()} 
                #tsne_df_merged = reduce(lambda  left,right: pd.merge(left,right,left_index=True,right_index=True,how='outer'),list(tsne_dict.values()))
                
                # Convert TSNE results to DF and reorganize columns
                tsne_df_merged=pd.concat(tsne_dict_run)
                tsne_df_merged=tsne_df_merged.reset_index().drop('level_1',axis=1)
                tsne_df_merged=tsne_df_merged.rename({'level_0':'tsneargs'},axis=1)


                # Data Type Label assignment
                num_lbls=[1 for i in range(0,405)]+[2] \
                +[3 for i in range(0,1200)]+[4]
                word_lbls=['wm_le' for i in range(0,405)]+['wm_sc'] \
                +['rest_le' for i in range(0,1200)]+['rest_sc']

                # Concatenate both PCA and TSNE
                #dim_red_df=pd.merge(pca_df,tsne_df_merged,left_index=True,right_index=True,how='outer')
                dim_red_df=pd.concat([pca_df,tsne_df_merged],sort=False)

                # Apply some labels to DF
                dim_red_df['Type']=word_lbls*4
                dim_red_df['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*4)
                dim_red_df['subid']=np.repeat(subname,1607*4)
                dim_red_df['method']=np.repeat(['pca','tsne','tsne','tsne'],1607)

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
    
    pca_comps=return_pca_comps(rest_wm_agg_wevs.T,n_components=3)
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

    # TSNE path to save unique runs
    tsne_dir=os.path.join('tsne_runs','59subs')
    if not os.path.isdir(tsne_dir):
        os.makedirs(tsne_dir)

    # TSNE of All
    #fast_tsne_LE_33=run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)
    fast_tsne_LE_59=run_multiple_fast_tsne(iplist,write_res=True,write_dir=tsne_dir)

    # Create tsne DF
    big_tsne_df_merged=pd.concat(fast_tsne_LE_59)
    big_tsne_df_merged=big_tsne_df_merged.reset_index().drop('level_1',axis=1)
    big_tsne_df_merged=big_tsne_df_merged.rename({'level_0':'tsneargs'},axis=1)


    x,y=big_tsne_df_merged.shape
    # Apply some labels to DF
    big_tsne_df_merged['VolumeAssignment']=np.reshape(np.repeat(np.vstack(np.arange(1,1608)),59*3,axis=1).T,[1607*59*3,1])
    big_tsne_df_merged['method']=np.repeat('tsne_59subs',x)
    big_tsne_df_merged['Type']=word_lbls*59*3
    big_tsne_df_merged['VolumeAssignment']=np.concatenate([np.arange(1,1608)]*59*3)
    big_tsne_df_merged['subid']=np.concatenate([np.repeat(bigdf.subid.unique(),1607)]*3)


        #print("Running HDBSCAN")
    #clus_sol=run_hdbscan(data_subset.T)


    subs=bigdf.subid.unique()
    # Merge big event df with dim red dfs
    fs=glob.glob('../../HCP-WM-LR-EPrime/*/*.csv')
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


    clus=run_hdbscan(rest_wm_agg_wevs)
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

            plt_opname='subs_66_plots/'+sub+'_dimred_'+hue+'.png'

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
