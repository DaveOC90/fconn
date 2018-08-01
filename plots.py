import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import io


def prior_dataproc():
    # Initial CV comparison plot
    cvdata=io.loadmat('./iteres.mat')
    cvdata=cvdata['res_struct']
    cvdata_extra=io.loadmat('./iteres_extra.mat')
    cvdata_extra=cvdata_extra['res_struct_extra']
    cvtypes=['k2','k5','k10','loo','external']
    cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']
    rpos=np.array([cvdata[0][cvt][0][:,0] for cvt in cvtypes]).T
    rpos_extra=np.array([cvdata_extra[0][cvt][0][:,0] for cvt in cvtypes]).T
    rposdf=pd.DataFrame(np.concatenate([rpos,rpos_extra]),columns=cvtypes_plots)

    cv_effect_plot()


    # Ntrainsubs plot
    nsubsdata=io.loadmat('./nsubs_iter.mat')
    train_lbls=['train'+str(num) for num in np.arange(25,401,25)]
    rpos=np.array([nsubsdata['res_struct_nsubs'][0][tl][0][:,1] for tl in train_lbls]).T
    nsubsdata_extra=io.loadmat('./nsubs_iter_extra.mat')
    pos_extra=np.array([nsubsdata_extra['res_struct_nsubs_extra'][0][tl][0][:,1] for tl in train_lbls]).T
    col_lbls=[str(num) for num in np.arange(25,401,25)]
    rposdf=pd.DataFrame(np.concatenate([rpos,rpos_extra]),columns=col_lbls)
    train_subs_plot()

    # Const train size plot
    ctrain_data=io.loadmat('./const_trainsize.mat')
    ctrain_data=ctrain_data['res_struct']
    cvtypes=['k2','k5','k10','loo','external']
    cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']

    rpos=np.array([ctrain_data[0][cvt][0][:,0] for cvt in cvtypes]).T

    rposdf=pd.DataFrame(rpos,columns=cvtypes_plots)
    const_trainsize_plot()


def run_cv_effect(fpath,structname,opname):
    cvdata=io.loadmat(fpath)

    cvdata=cvdata[structname]

    cvtypes=['k2','k5','k10','loo','external']
    
    cvtypes_plots=['Split Half', '5 Fold', '10 Fold', 'LOO', 'External']
    
    rpos=np.array([cvdata[0][cvt][0][:,0] for cvt in cvtypes]).T

    ipdata=pd.DataFrame(rpos,columns=cvtypes_plots)
    #ipdata=ipdata[['Split Half', '5 Fold', '10 Fold', 'LOO']]

    cv_effect_plot(ipdata,opname)

def cv_effect_plot(ipdata,opname):
    plt.clf()
    sns.set(style='ticks')
    sns.boxplot(data=ipdata,palette='vlag')
    #sns.violinplot(data=rposdf,palette='vlag',inner='quartile')
    sns.despine(offset=10, trim=True)
    plt.title('Effect of CV Method on Prediction - Variable Train Size',fontsize='large')
    plt.ylabel('MSE',weight='bold')
    plt.xlabel('Cross Validation Method',weight='bold')
    plt.tight_layout()
    #plt.savefig('./cv_pred_200.jpg', dpi=None, facecolor='w', edgecolor='w',
    #    orientation='portrait', papertype=None, format=None,
    #    transparent=False, bbox_inches=None, pad_inches=0.1,
    #    frameon=None)
    plt.savefig(opname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)    

def train_subs_plot():
    plt.clf()
    sns.set(style='ticks')
    sns.boxplot(data=rposdf,palette='vlag')
    #sns.violinplot(data=rposdf,palette='vlag',inner='quartile')
    sns.despine(offset=10, trim=True)
    plt.title('Effect of number of Training Subjects on Prediction',fontsize='large')
    plt.ylabel('R Value',weight='bold')
    plt.xlabel('Number of Training Subjects',weight='bold')
    plt.tight_layout()
    plt.savefig('./nsubs_pred_200.jpg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)


def const_trainsize_plot():
    plt.clf()
    sns.set(style='ticks')
    sns.boxplot(data=rposdf,palette='vlag')
    #sns.violinplot(data=rposdf,palette='vlag',inner='quartile')
    sns.despine(offset=10, trim=True)
    plt.title('Effect of CV Method on Prediction - Constant Train Size (180)',fontsize='large')
    plt.ylabel('R Value',weight='bold')
    plt.xlabel('Cross Validation Method',weight='bold')
    plt.tight_layout()
    plt.savefig('./cv_pred_const_train.jpg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)


