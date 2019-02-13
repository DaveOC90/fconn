import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import io
from matplotlib.colors import ListedColormap
from sklearn import linear_model

data_dir='C:/Users/david/Documents/Research/10simplerules_data_figs/data'
fig_dir='C:/Users/david/Documents/Research/10simplerules_data_figs/figs'


def nsubs_plot():
    # Ntrainsubs plot
    nsubsdata=io.loadmat(data_dir+'/norm_nsubtrain.mat')
    nsubsdata=nsubsdata['behav_struct_nsubs']
    
    train_lbls=['train'+str(num) for num in np.arange(25,401,25)]    
    col_lbls=[str(num) for num in np.arange(25,401,25)]

    #rpos=np.array([nsubsdata['res_struct_nsubs'][0][tl][0][:,1] for tl in train_lbls]).T
    
    # Aggregate predicted behavior-
    pred_data_dict={cvt:nsubsdata[0][cvt][0]['predbehavpos'][0][0] for cvt in train_lbls}
    real_data_dict={cvt:nsubsdata[0]['testbehav'][0] for cvt in train_lbls}
    

    cllct=[]
    for cvt in train_lbls:
    
        mse=np.mean(np.array(pred_data_dict[cvt]-real_data_dict[cvt])**2,axis=1)
        cllct.append(mse)

    mse_tot=np.array(cllct)
    
    ipdata=pd.DataFrame(mse_tot.T,columns=col_lbls)


    #rposdf=pd.DataFrame(rpos,columns=col_lbls)
    
    train_subs_plot(ipdata, fig_dir+'/nsubs_mse.tiff')


def sametrain_run():
    # Const train size plot
    ctrain_data=io.loadmat(data_dir+'/sametrain_norm.mat')
    #cvdata=ctrain_data['res_struct_norm_st']
    cvdata=ctrain_data['behav_struct_norm_st']
    #cvtypes=['k2','k5','k10','loo','external']
    #cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']
 
    cvtypes=['k2','k5','k10','loo']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO']

    #rpos=np.array([ctrain_data[0][cvt][0][:,0] for cvt in cvtypes]).T

    #rposdf=pd.DataFrame(rpos,columns=cvtypes_plots)

    pncpmat=io.loadmat(data_dir+"/pncpmats.mat",variable_names=['pmats_pnc'])
    pncpmat=pncpmat['pmats_pnc'][0]
    pncpmat=(pncpmat-pncpmat.mean())/pncpmat.std()

    # Aggregate predicted behavior-
    pred_data_dict={cvt:cvdata[0][cvt][0]['predbehavpos'][0][0] for cvt in cvtypes}
    real_data_dict={cvt:cvdata[0][cvt][0]['testbehav'][0][0] for cvt in cvtypes}
    real_data_dict['external']=np.repeat(np.vstack(pncpmat).T,100,axis=0)


    cllct=[]
    for cvt in cvtypes:
    
        mse=np.mean(np.array(pred_data_dict[cvt]-real_data_dict[cvt])**2,axis=1)
        cllct.append(mse)

    mse_tot=np.array(cllct)
    
    ipdata=pd.DataFrame(mse_tot.T,columns=cvtypes_plots)

    ipdata=ipdata[['Split-half', '5-fold', '10-fold', 'LOO']]

    const_trainsize_plot(ipdata,fig_dir+'/sametrain_mse.tiff')

def mseplot():
    # Load Data
    cvdata=io.loadmat(data_dir+'/norm_cvcomparison_cuda.mat')
    cvdata=cvdata['behav_struct_norm']
    cvdata_extra=io.loadmat(data_dir+'/norm_cvcomparison_shred.mat')
    cvdata_extra=cvdata_extra['behav_struct_norm']
    pncpmat=io.loadmat(data_dir+"/pncpmats.mat",variable_names=['pmats_pnc'])
    pncpmat=pncpmat['pmats_pnc'][0]
    pncpmat=(pncpmat-pncpmat.mean())/pncpmat.std()
    # Set labels
    cvtypes=['k2','k5','k10','loo','external']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO', 'External']
    cvtypes_int=['k2','k5','k10','loo']
    
    # Aggregate predicted behavior-
    pred_data_dict={cvt:cvdata[0][cvt][0]['predbehavpos'][0][0] for cvt in cvtypes}
    real_data_dict={cvt:cvdata[0][cvt][0]['testbehav'][0][0] for cvt in cvtypes_int}
    real_data_dict['external']=np.repeat(np.vstack(pncpmat).T,100,axis=0)


    cllct=[]
    for cvt in cvtypes:

        mse=np.mean(np.array(pred_data_dict[cvt]-real_data_dict[cvt])**2,axis=1)
        cllct.append(mse)

    mse_tot=np.array(cllct)
    
    ipdata=pd.DataFrame(mse_tot.T,columns=cvtypes_plots)

    ipdata=ipdata[['Split-half', '5-fold', '10-fold', 'LOO']]

    cv_effect_plot(ipdata,fig_dir+'/normcvMSE.tiff')
    #cv_effect_plot_sepscales(ipdata,'../normcvMSE_splitscale.jpg')

def cv_effect_plot(ipdata,opname,title):
    plt.clf()
    sns.set(style='ticks')
    #sns.boxplot(data=ipdata[['Split Half', '5 Fold', '10 Fold', 'LOO']],palette='vlag')
    sns.boxplot(data=ipdata,palette='vlag')
    plt.ylim([0,0.5])
    #sns.violinplot(data=rposdf,palette='vlag',inner='quartile')
    sns.despine(offset=10, trim=True)
    plt.title(title,fontsize='large')
    plt.ylabel('R',weight='bold')
    plt.xlabel('Cross-validation method',weight='bold')
    plt.tight_layout()
    #plt.savefig('./cv_pred_200.jpg', dpi=None, facecolor='w', edgecolor='w',
    #    orientation='portrait', papertype=None, format=None,
    #    transparent=False, bbox_inches=None, pad_inches=0.1,
    #    frameon=None)
    plt.savefig(opname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)



def train_subs_plot(ipdata,opname):
    plt.clf()
    sns.set(style='ticks')
    sns.boxplot(data=ipdata,palette='vlag')
    #sns.violinplot(data=rposdf,palette='vlag',inner='quartile')
    sns.despine(offset=10, trim=True)
    plt.title('Effect of number of training individuals on prediction',fontsize='large')
    plt.ylabel('MSE',weight='bold')
    plt.xlabel('Number of training individuals',weight='bold')
    plt.tight_layout()
    plt.savefig(opname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)


def const_trainsize_plot(ipdata,opname):
    plt.clf()
    sns.set(style='ticks')
    sns.boxplot(data=ipdata,palette='vlag')
    #sns.violinplot(data=rposdf,palette='vlag',inner='quartile')
    sns.despine(offset=10, trim=True)
    plt.title('Constant train size (n=180)',fontsize='large')
    plt.ylabel('MSE',weight='bold')
    plt.xlabel('Cross-validation method',weight='bold')
    plt.tight_layout()
    plt.savefig(opname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)


#### WIP ####

def run_cv_effect(fpath,structname,opname):
    cvdata=io.loadmat(fpath)

    cvdata=cvdata[structname]

    cvtypes=['k2','k5','k10','loo','external']
    
    cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']
    
    rpos=np.array([cvdata[0][cvt][0][:,0] for cvt in cvtypes]).T

    ipdata=pd.DataFrame(rpos,columns=cvtypes_plots)
    #ipdata=ipdata[['Split Half', '5 Fold', '10 Fold', 'LOO']]

    cv_effect_plot(ipdata,opname)



def prior_dataproc():
    # Initial CV comparison plot
    cvdata=io.loadmat('./iteres.mat')
    cvdata=cvdata['res_struct']
    #cvdata_extra=io.loadmat('./iteres_extra.mat')
    #cvdata_extra=cvdata_extra['res_struct_extra']
    cvtypes=['k2','k5','k10','loo','external']
    cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']
    rpos=np.array([cvdata[0][cvt][0][:,0] for cvt in cvtypes]).T
    #rpos_extra=np.array([cvdata_extra[0][cvt][0][:,0] for cvt in cvtypes]).T
    #rposdf=pd.DataFrame(np.concatenate([rpos,rpos_extra]),columns=cvtypes_plots)
    rposdf=pd.DataFrame(np.concatenate([rpos,rpos_extra]),columns=cvtypes_plots)
    cv_effect_plot()





def quickcorrplot(ip_fpath,op_plot_name,title):
    # Initial CV comparison plot
    
    cvdata=io.loadmat(ip_fpath)
    cvdata=cvdata['res_struct']

    #cvtypes=['k2','k5','k10','loo','external']
    #cvtypes=['k2','k5','k10','loo']
    cvtypes=['folds_2','folds_5','folds_10','folds_500']
    
    #cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO']
    
    rpos=np.array([cvdata[0][cvt][0][:,0] for cvt in cvtypes]).T
    #rpos_extra=np.array([cvdata_extra[0][cvt][0][:,0] for cvt in cvtypes]).T
    #rposdf=pd.DataFrame(np.concatenate([rpos,rpos_extra]),columns=cvtypes_plots)
    rposdf=pd.DataFrame(rpos,columns=cvtypes_plots)

    cv_effect_plot(rposdf,op_plot_name,title)

def quickcorrplot(ip_fpath,op_plot_name,title):
    # Initial CV comparison plot
    
    cvdata=io.loadmat(ip_fpath)
    cvdata=cvdata['res_struct']

    #cvtypes=['k2','k5','k10','loo','external']
    #cvtypes=['k2','k5','k10','loo']
    cvtypes=['folds_2','folds_5','folds_10','folds_500']
    
    #cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO']
    
    rpos=np.array([cvdata[0][cvt][0][:,0] for cvt in cvtypes]).T
    #rpos_extra=np.array([cvdata_extra[0][cvt][0][:,0] for cvt in cvtypes]).T
    #rposdf=pd.DataFrame(np.concatenate([rpos,rpos_extra]),columns=cvtypes_plots)
    rposdf=pd.DataFrame(rpos,columns=cvtypes_plots)

    cv_effect_plot(rposdf,op_plot_name,title)
    

def multiboxplot():
    mats=glob.glob('../LEiDA/data/gsrcomp/*')
    nums={m.split('\\')[-1]:io.loadmat(m)['res_struct'] for m in mats}

    cvtypes1=['k2','k5','k10','loo']
    cvtypes2=['folds_2','folds_5','folds_10','folds_500']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO']

    dflist=[]
    for nk in nums.keys():
        try:
            rpos=np.array([nums[nk][0][cvt][0][:,0] for cvt in cvtypes1]).T
        except ValueError:
            rpos=np.array([nums[nk][0][cvt][0][:,0] for cvt in cvtypes2]).T

        if 'scfisher' in nk:
            mattype='scfisher'
        elif '_le2' in nk:
            mattype='LE2'
        elif '_le' in nk:
            mattype='LE'

        
        if 'gsr' in nk:
            gsr='gsr'
        else:
            gsr='nogsr'

        if 'rest' in nk:
            state='rest'
        elif 'wm' in nk:
            state='wm'
        else:
            raise Exception('NOSTATEEEE')

        if rpos.shape[0] > 50:
            rpos=rpos[:50,:]

        cvcats=[[l,gsr,mattype,state] for l in list(np.reshape([cvtypes2]*50,200,1))]
        rpos=np.reshape(rpos,200)
  
        df=pd.DataFrame(cvcats,columns=['CV','GSR','MatType','State'])
        df['RPos']=rpos

        dflist.append(df)

    alldata=pd.concat(dflist)


def cv_effect_plot_sepscales(ipdata,opname):
    plt.clf()


    #fig.set_ylabel('R',weight='bold')
    #fig.set_xlabel('Cross Validation Method',weight='bold')
    #my_cmap=ListedColormap(sns.color_palette('vlag').as_hex())
    my_cmap=sns.color_palette('vlag').as_hex()[0:5]

    f,a1=plt.subplots()
    sns.set_palette("vlag")

    props = dict(widths=0.7,patch_artist=True, medianprops=dict(color="black"))

    b1=a1.boxplot(ipdata[['Split Half', '5 Fold', '10 Fold', 'LOO']].values,positions=[0,1,2,3],**props)

    a2=a1.twinx()
    b2=a2.boxplot(ipdata[['External']].values,positions=[4],**props)
    
    a1.set_xlim(-0.5,4.5)
    a1.set_xticks(range(len(ipdata.columns)))
    a1.set_xticklabels(ipdata.columns)

    for patch, color in zip(b1['boxes']+b2['boxes'], my_cmap):
        patch.set_facecolor(color)

    sns.despine(offset=10, trim=True,right=False)
    sns.set(style='ticks')

    plt.suptitle('Variable train size',fontsize='large')
    a1.set_ylabel('MSE',weight='bold')
    a1.set_xlabel('Cross-validation method',weight='bold')

    plt.tight_layout()

    plt.savefig(opname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)    
