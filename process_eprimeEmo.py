import pandas as pd 
import glob
import numpy as np
from collections import Counter
import operator
import os
import sys
import pdb
import natsort



eprimePath = sys.argv[1]

alleprimefpaths=natsort.natsorted(glob.glob(os.path.join(eprimePath,'*/*/*EMOTION*.txt')))




cols_of_interest = ['shape.','StimSlide.', 'Fixation.','face.']




index=pd.MultiIndex.from_tuples([(c.split('.')) for c in cols_of_interest],names=['Event','Property'])

for aef in alleprimefpaths:
    print(aef)

    df_tst=pd.read_csv(aef,sep='\t')
    sub_coi=[c for c in df_tst.columns if any(x in c for x in cols_of_interest)]
    df_tst=df_tst[sub_coi]
    index=pd.MultiIndex.from_tuples([(c.split('.')) for c in sub_coi],names=['Event','Property'])
    df_multi=pd.DataFrame(df_tst.values,columns=index)
    stackedinfo=df_multi.swaplevel(axis=1).stack()
    onsets_corr=stackedinfo.OnsetTime-stackedinfo.OnsetTime.values[0]
    RTonsets_corr=stackedinfo.RTTime-stackedinfo.OnsetTime.values[0]
    stackedinfo['CorrectedOnsetTime']=onsets_corr
    stackedinfo['CorrectedRTTime']=RTonsets_corr
    volume_timing=[720*i for i in range(0,176)]
    stackedinfo['VolumeAssignment']=np.digitize(onsets_corr.astype('int32'),volume_timing)
    stackedinfo['EventCol']=stackedinfo.index.to_frame(index=True)['Event'].values

    #stackedinfo.EventCol[stackedinfo.EventCol == 'Stim']= stackedinfo[stackedinfo.EventCol == 'StimSlide'][['EventCol','BlockType','StimType']].apply(lambda x: '-'.join(x),axis=1)


    stackedinfo.to_csv(aef.replace('.txt','_filtered.csv'))




fs=natsort.natsorted(glob.glob(os.path.join(eprimePath,'*/*/*filt*')))
df_list=[pd.read_csv(f,index_col=0) for f in fs]

sublist=[f.split('/')[-3] for f in fs]

nsubs = len(sublist)


aggeventcols=pd.concat([df.EventCol for df in df_list],axis=1) 
aggeventcols.columns=sublist 
aggeventcols.apply(Counter,axis=1)
compareres=pd.concat([aggeventcols.apply(lambda x: max(Counter(x).keys(),key=operator.itemgetter(1)),axis=1),aggeventcols.apply(lambda x: Counter(x),axis=1)],axis=1)
mostcommon=aggeventcols.apply(lambda x: max(Counter(x).keys(),key=operator.itemgetter(1)),axis=1)
subswithsamedata=[k for k in aggeventcols.columns if np.sum(aggeventcols[k]==mostcommon) == 72]




flist=natsort.natsorted(glob.glob(os.path.join(eprimePath,'*/*/*_3T_EMOTION_run*_TAB_filtered.csv')))
dflist=[pd.read_csv(f,index_col=0) for f in flist]

accarr=np.zeros([176,nsubs])
accarr[:,:]=np.nan
for i,df in enumerate(dflist):
    accarr[df.VolumeAssignment.values-1,i]=df.ACC.values



RTarr=np.zeros([176,nsubs])
RTarr[:,:]=np.nan
for i,df in enumerate(dflist):
    print(flist[i])
    RTarr[df.VolumeAssignment.values-1,i]=df.RT.values




#sublist=list(map(lambda x: 'sub'+x.split('/')[-3],flist)) 




#RTarr[np.isnan(RTarr)]=0
RTdf=pd.DataFrame(RTarr)
RTdf=RTdf.replace(to_replace=np.nan,method='ffill',limit=3)
RTdf.columns=sublist 
RTdf.to_csv(os.path.join(eprimePath,'ResponseTimeByTP.csv'))


'''
#accarr[np.isnan(accarr)]=0
accdf=pd.DataFrame(accarr)
accdf=accdf.replace(to_replace=np.nan,method='ffill',limit=3)
accdf.columns=sublist 
accdf.to_csv(os.path.join(eprimePath,'AccuracyByTP.csv'))



RTdf_ffill=RTdf.replace(to_replace=np.nan,method='ffill',limit=3)

problem_tps=RTdf_ffill.loc[np.sum(RTdf_ffill == 0,axis=1) > 50]
substokeep=problem_tps[~(problem_tps == 0)].dropna(axis=1).columns


np.save(os.path.join(eprimePath,'subswithlesszeros.npy'),substokeep)

eventcolarr=np.zeros([405,nsubs],'object_')
for i,df in enumerate(df_list):
    eventcolarr[df.VolumeAssignment.values-1,i]=df.EventCol.values
ECdf=pd.DataFrame(eventcolarr)
ECdf.columns=sublist
ECdf.to_csv(os.path.join(eprimePath,'EventColbySub.csv'))




eventdf=ECdf
eventdf_ffill=eventdf.replace(to_replace=0,method='ffill')
eventdf_ffill_nofix=eventdf_ffill.replace(to_replace='Fix',method='ffill')

ZeroBackMaskdf=eventdf_ffill_nofix.apply(lambda x : ['0-Back' in x1 for x1 in x],axis=0)
TwoBackMaskdf=eventdf_ffill_nofix.apply(lambda x : ['2-Back' in x1 for x1 in x],axis=0)

ZeroBackMaskdf.columns=list(map(lambda x:x.replace('sub',''),ZeroBackMaskdf.columns))
TwoBackMaskdf.columns=list(map(lambda x:x.replace('sub',''),TwoBackMaskdf.columns))

ZeroBackMaskdf.to_csv(os.path.join(eprimePath,'zerobackmask.csv'))
TwoBackMaskdf.to_csv(os.path.join(eprimePath,'twobackmask.csv'))

ZeroandTwo=ZeroBackMaskdf | TwoBackMaskdf
NoStim=~ZeroandTwo
NoStim.to_csv(os.path.join(eprimePath,'restmask.csv'))


for i in range(0,4):
    startindex=i*2
    endindex=(i*2)+1
    
    print(startindex,endindex)

    ZeroFalseArr=np.zeros(ZeroBackMaskdf.values.shape).astype(bool)
    ZeroBlockIndices=ZeroBackMaskdf.apply(lambda x : np.nonzero(np.diff(x))[0],axis=0)
    for j,col in enumerate(ZeroBlockIndices):
        subindices=ZeroBlockIndices[col]
        # Might need to add one to these indices
        ZeroFalseArr[subindices[startindex]:subindices[endindex],j]=ZeroBackMaskdf[col].values[subindices[startindex]:subindices[endindex]]

    ZeroBackMaskdfFalse=pd.DataFrame(ZeroFalseArr,columns=ZeroBackMaskdf.columns)
    ZeroBackMaskdfFalse.to_csv(os.path.join(eprimePath,'ZeroBackMaskBlock'+str(i+1)+'.csv'))



    TwoFalseArr=np.zeros(TwoBackMaskdf.values.shape).astype(bool)
    TwoBlockIndices=TwoBackMaskdf.apply(lambda x : np.nonzero(np.diff(x))[0],axis=0)
    for j,col in enumerate(TwoBlockIndices):
        subindices=TwoBlockIndices[col]
        # Might need to add one to these indices
        TwoFalseArr[subindices[startindex]:subindices[endindex],j]=TwoBackMaskdf[col].values[subindices[startindex]:subindices[endindex]]

    TwoBackMaskdfFalse=pd.DataFrame(TwoFalseArr,columns=TwoBackMaskdf.columns)
    TwoBackMaskdfFalse.to_csv(os.path.join(eprimePath,'TwoBackMaskBlock'+str(i+1)+'.csv'))


'''

