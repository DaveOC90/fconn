import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from nilearn import plotting
import nibabel as nib
import subprocess
from PIL import Image

import pdb
import sys
import os

import chordplot_edit


def make_montage(edges,opname):
    square_edges=np.reshape(edges,[268,268])
    edgedf=pd.DataFrame(square_edges)
    colnames=['node'+str(i).zfill(3) for i in range(1,269)]
    edgedf.index=colnames
    edgedf.columns=colnames


    # Network mapping and sorting
    netmap=pd.read_csv('/home/dmo39/map268_subnetwork.csv')
    nodesortlist='node'+netmap.oldroi.astype(str).str.zfill(3)

    edgedf = edgedf[nodesortlist].loc[nodesortlist] 


    # Go to long form
    trimask=np.triu(np.ones(edgedf.shape),k=1).astype(bool)
    edgedf_tri=edgedf.where(trimask)
    edgedf_stack=edgedf_tri.stack().reset_index()
    edgedf_stack.columns=['source','dest','weight']
    edgedf_stack['sourceid']=edgedf_stack.source 
    edgedf_stack['destid']=edgedf_stack.dest                


    # Create new id mapping to sort chord plot
    idmapping=dict(zip('node'+netmap.oldroi.astype(str).str.zfill(3),netmap.newroi.values)) 

    edgedf_stack.replace({'destid':idmapping},inplace=True)
    edgedf_stack.replace({'sourceid':idmapping},inplace=True)


    edgedf_stack['SourceNetwork']=edgedf_stack.sourceid
    edgedf_stack['DestNetwork']=edgedf_stack.destid

    netmapdict=dict(zip(netmap.newroi,netmap.label))
    edgedf_stack.replace({'SourceNetwork':netmapdict},inplace=True)
    edgedf_stack.replace({'DestNetwork':netmapdict},inplace=True)


    # Network Level DF
    edgedf_stack_net=edgedf_stack[['SourceNetwork','DestNetwork','weight']]
    netsumdf=edgedf_stack_net.groupby(['SourceNetwork','DestNetwork']).agg('sum').unstack()
    netsumdf.values[~np.isnan(netsumdf.values.T)]=netsumdf.values.T[~np.isnan(netsumdf.values.T)]

    mask = np.zeros_like(netsumdf, dtype=np.bool)
    mask[np.triu_indices_from(mask,k=1)] = True


    # Plot sum of network level presence in edge permutation
    netsumdf.columns=netsumdf.columns.droplevel()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    opname_netsum=opname.split('.')[0]+'_netsummat.png'

    plt.figure(figsize=[6.5,4.875])
    sns.heatmap(netsumdf,annot=True,mask=mask,cmap=cmap)
    plt.tight_layout()
    plt.savefig(opname_netsum)

    # Histogram of weights, not very useful
    wvals=edgedf_stack.weight.values
    wvals_nonzero=wvals[wvals!=0]

    #sns.distplot(wvals_nonzero)
    #plt.show()


    # Plot degree centrality
    parcelFile=nib.Nifti1Image.load('/home/dmo39/shen_2mm_268_parcellation.nii.gz')
    parcelData=parcelFile.get_data()

    #imfile='/home/dmo39/shen_2mm_268_parcellation.nii.gz'
    #plotting.plot_glass_brain(imfile,colorbar=True)
    #plt.show()


    degreeSeries=edgedf.sum()
    degreeData=parcelData

    for i in range(1,269):
       degreenum=degreeSeries['node'+str(i).zfill(3)]
       degreeData[degreeData == i] = degreenum

    degreeObj=nib.Nifti1Image(degreeData,parcelFile.affine)
    opPathDC=opname.split('.')[0]+'_degreeCentrality.nii.gz'
    degreeObj.to_filename(opPathDC)


    fig=plt.figure(figsize=[8,2])
    plotting.plot_glass_brain(opPathDC,colorbar=True,display_mode='lzry',figure=fig)


    opPathDCPlot=opname.split('.')[0]+'_degreeCentrality.png'
    plt.savefig(opPathDCPlot)

    opPathChordPlot=opname.split('.')[0]+'_chordplot.png'

    chordplot_edit.run_main(edgedf_stack,netmap,opPathChordPlot)

    new_im=Image.new('RGB',(1600,850),color=(255,255,255))
    im1=Image.open(opPathChordPlot)
    im2=Image.open(opPathDCPlot)
    im3=Image.open(opname_netsum)

    new_im.paste(im1)
    new_im.paste(im2,(800,600))
    new_im.paste(im3,(900,100))


    new_im.save(opname)





def make_montage_diffhem(edges,opname):
    square_edges=np.reshape(edges,[268,268])
    edgedf=pd.DataFrame(square_edges)
    colnames=['node'+str(i).zfill(3) for i in range(1,269)]
    edgedf.index=colnames
    edgedf.columns=colnames


    # Network mapping and sorting
    netmap=pd.read_csv('/home/dmo39/map268_subnetwork.csv')

    netmap = netmap.sort_values(by = ['hemisphere','category'])    

    netmapDiffHem1 = netmap.iloc[:135]
    netmapDiffHem2 = netmap.iloc[135:]

    netmapDiffHem2 = netmapDiffHem2.sort_values(by = ['category'],ascending=False)

    netmapDiffHem = netmapDiffHem1.append(netmapDiffHem2) #netmap.sort_values(by = ['hemisphere','category'])
    nodesortlist='node'+netmapDiffHem.oldroi.astype(str).str.zfill(3)

    edgedf = edgedf[nodesortlist].loc[nodesortlist]

    # Go to long form
    trimask=np.triu(np.ones(edgedf.shape),k=1).astype(bool)
    edgedf_tri=edgedf.where(trimask)
    edgedf_stack=edgedf_tri.stack().reset_index()
    edgedf_stack.columns=['source','dest','weight']
    edgedf_stack['sourceid']=edgedf_stack.source 
    edgedf_stack['destid']=edgedf_stack.dest                


    # Create new id mapping to sort chord plot
    idmapping=dict(zip('node'+netmapDiffHem.oldroi.astype(str).str.zfill(3),np.array(range(1,269))))#netmap.newroi.values)) 

    edgedf_stack.replace({'destid':idmapping},inplace=True)
    edgedf_stack.replace({'sourceid':idmapping},inplace=True)


    edgedf_stack['SourceNetwork']=edgedf_stack.sourceid
    edgedf_stack['DestNetwork']=edgedf_stack.destid

    #netmapdict=dict(zip(netmapDiffHem.newroi,netmapDiffHem.label))
    netmapdict=dict(zip(range(1,269),netmapDiffHem.label))
    edgedf_stack.replace({'SourceNetwork':netmapdict},inplace=True)
    edgedf_stack.replace({'DestNetwork':netmapdict},inplace=True)


    # Network Level DF
    edgedf_stack_net=edgedf_stack[['SourceNetwork','DestNetwork','weight']]
    netsumdf=edgedf_stack_net.groupby(['SourceNetwork','DestNetwork']).agg('sum').unstack()
    netsumdf.values[~np.isnan(netsumdf.values.T)]=netsumdf.values.T[~np.isnan(netsumdf.values.T)]


    mask = np.zeros_like(netsumdf, dtype=np.bool)
    mask[np.triu_indices_from(mask,k=1)] = True


    # Plot sum of network level presence in edge permutation
    netsumdf.columns=netsumdf.columns.droplevel()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    opname_netsum=opname.split('.')[0]+'_netsummat.png'

    plt.figure(figsize=[6.5,4.875])
    sns.heatmap(netsumdf,annot=True,mask=mask,cmap=cmap)
    plt.tight_layout()
    plt.savefig(opname_netsum)

    # Histogram of weights, not very useful
    wvals=edgedf_stack.weight.values
    wvals_nonzero=wvals[wvals!=0]

    #sns.distplot(wvals_nonzero)
    #plt.show()


    # Plot degree centrality
    parcelFile=nib.Nifti1Image.load('/home/dmo39/shen_2mm_268_parcellation.nii.gz')
    parcelData=parcelFile.get_data()

    #imfile='/home/dmo39/shen_2mm_268_parcellation.nii.gz'
    #plotting.plot_glass_brain(imfile,colorbar=True)
    #plt.show()


    degreeSeries=edgedf.sum()
    degreeData=parcelData

    for i in range(1,269):
       degreenum=degreeSeries['node'+str(i).zfill(3)]
       degreeData[degreeData == i] = degreenum

    degreeObj=nib.Nifti1Image(degreeData,parcelFile.affine)
    opPathDC=opname.split('.')[0]+'_degreeCentrality.nii.gz'
    degreeObj.to_filename(opPathDC)


    fig=plt.figure(figsize=[8,2])
    plotting.plot_glass_brain(opPathDC,colorbar=True,display_mode='lzry',figure=fig)


    opPathDCPlot=opname.split('.')[0]+'_degreeCentrality.png'
    plt.savefig(opPathDCPlot)

    opPathChordPlot=opname.split('.')[0]+'_chordplot.png'

    chordplot_edit.run_main_diffhem(edgedf_stack,netmapDiffHem,opPathChordPlot)

    new_im=Image.new('RGB',(1600,850),color=(255,255,255))
    im1=Image.open(opPathChordPlot)
    im2=Image.open(opPathDCPlot)
    im3=Image.open(opname_netsum)

    new_im.paste(im1)
    new_im.paste(im2,(800,600))
    new_im.paste(im3,(900,100))


    new_im.save(opname)



if __name__ == '__main__':
    resfile=sys.argv[1]
    opdir=sys.argv[2]


    print("Prepping Input")

    #resfile='/data15/mri_group/dave_data/dcpm/staticcon_fIQ_400_400_0back/results/dCPM_static_results_test_0back.npy'

    filename=os.path.basename(resfile)

    # Read example model edges and convert to DF
    zerobackres=np.load(resfile,allow_pickle=True)
    square_edges=np.reshape(zerobackres[0]['posedges'],[268,268])
    edgedf=pd.DataFrame(square_edges)
    colnames=['node'+str(i).zfill(3) for i in range(1,269)]
    edgedf.index=colnames
    edgedf.columns=colnames


    # Network mapping and sorting
    netmap=pd.read_csv('/home/dmo39/map268_subnetwork.csv')
    nodesortlist='node'+netmap.oldroi.astype(str).str.zfill(3)
    edgedf = edgedf[nodesortlist].ix[nodesortlist] 


    # Go to long form
    trimask=np.triu(np.ones(edgedf.shape),k=1).astype(bool)
    edgedf_tri=edgedf.where(trimask)
    edgedf_stack=edgedf_tri.stack().reset_index()
    edgedf_stack.columns=['source','dest','weight']
    edgedf_stack['sourceid']=edgedf_stack.source 
    edgedf_stack['destid']=edgedf_stack.dest                


    # Create new id mapping to sort chord plot
    idmapping=dict(zip('node'+netmap.oldroi.astype(str).str.zfill(3),netmap.newroi.values)) 

    edgedf_stack.replace({'destid':idmapping},inplace=True)
    edgedf_stack.replace({'sourceid':idmapping},inplace=True)


    edgedf_stack['SourceNetwork']=edgedf_stack.sourceid
    edgedf_stack['DestNetwork']=edgedf_stack.destid

    netmapdict=dict(zip(netmap.newroi,netmap.label))
    edgedf_stack.replace({'SourceNetwork':netmapdict},inplace=True)
    edgedf_stack.replace({'DestNetwork':netmapdict},inplace=True)


    # Network Level DF
    edgedf_stack_net=edgedf_stack[['SourceNetwork','DestNetwork','weight']]
    netsumdf=edgedf_stack_net.groupby(['SourceNetwork','DestNetwork']).agg('sum').unstack()
    netsumdf.values[~np.isnan(netsumdf.values.T)]=netsumdf.values.T[~np.isnan(netsumdf.values.T)]

    mask = np.zeros_like(netsumdf, dtype=np.bool)
    mask[np.triu_indices_from(mask,k=1)] = True


    # Plot sum of network level presence in edge permutation
    netsumdf.columns=netsumdf.columns.droplevel()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    pltname=filename.split('.')[0]+'_netsummat.png'
    opname_netsum=os.path.join(opdir,pltname)

    plt.figure(figsize=[6.5,4.875])
    sns.heatmap(netsumdf,annot=True,mask=mask,cmap=cmap)
    plt.tight_layout()
    plt.savefig(opname_netsum)

    # Histogram of weights, not very useful
    wvals=edgedf_stack.weight.values
    wvals_nonzero=wvals[wvals!=0]

    #sns.distplot(wvals_nonzero)
    #plt.show()


    # Plot degree centrality
    parcelFile=nib.Nifti1Image.load('/home/dmo39/shen_2mm_268_parcellation.nii.gz')
    parcelData=parcelFile.get_data()

    #imfile='/home/dmo39/shen_2mm_268_parcellation.nii.gz'
    #plotting.plot_glass_brain(imfile,colorbar=True)
    #plt.show()


    degreeSeries=edgedf.sum()
    degreeData=parcelData

    for i in range(1,269):
       degreenum=degreeSeries['node'+str(i).zfill(3)]
       degreeData[degreeData == i] = degreenum

    degreeObj=nib.Nifti1Image(degreeData,parcelFile.affine)
    DCname=filename.split('.')[0]+'_degreeCentrality.nii.gz'
    opPathDC=os.path.join(opdir,DCname)
    degreeObj.to_filename(opPathDC)


    fig=plt.figure(figsize=[8,2])
    plotting.plot_glass_brain(opPathDC,colorbar=True,display_mode='lzry',figure=fig)

    DCPlotName=filename.split('.')[0]+'_degreeCentrality.png'
    opPathDCPlot=os.path.join(opdir,DCPlotName)
    plt.savefig(opPathDCPlot)

    ChordPlotName=filename.split('.')[0]+'_chordplot.png'
    opPathChordPlot=os.path.join(opdir,ChordPlotName)

    chordplot_edit.run_main(edgedf_stack,netmap,opPathChordPlot)

    new_im=Image.new('RGB',(1600,850),color=(255,255,255))
    im1=Image.open(opPathChordPlot)
    im2=Image.open(opPathDCPlot)
    im3=Image.open(opname_netsum)

    new_im.paste(im1)
    new_im.paste(im2,(800,600))
    new_im.paste(im3,(900,100))


    MontagePlotName=filename.split('.')[0]+'_montage.png'
    opPathMontagePlot=os.path.join(opdir,MontagePlotName)
    new_im.save(opPathMontagePlot)



