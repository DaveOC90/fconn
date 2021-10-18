import pandas as pd 
import glob
import numpy as np
from collections import Counter
import operator
import os
import sys


eprimePath = sys.argv[1]

alleprimefpaths=glob.glob(os.path.join(eprimePath,'*/*WM*.txt'))


cols_of_interest=[
 'Cue2Back.',
 'Stim.',
 'Fix.',
 'CueTarget.',
 'Fix15sec.',
 'SyncSlide.',
 'TargetType',
 'StimType',
 'BlockType']

change_names={
	'TargetType':'Stim.TargetType',
	'StimType':'Stim.StimType',
	'BlockType':'Stim.BlockType'
}

#index=pd.MultiIndex.from_tuples([(c.split('.')) for c in cols_of_interest],names=['Event','Property'])

for aef in alleprimefpaths:
    print(aef)
    df_tst=pd.read_csv(aef,sep='\t')
    try:
        df_tst=df_tst.rename(index=str,columns=change_names)
        
    except KeyError:
        raise('At least one column to be renamed doesnt exist!!!')

    sub_coi=[c for c in df_tst.columns if any(x in c for x in cols_of_interest)]
    df_tst=df_tst[sub_coi]
    index=pd.MultiIndex.from_tuples([(c.split('.')) for c in sub_coi],names=['Event','Property'])
    df_multi=pd.DataFrame(df_tst.values,columns=index)
    stackedinfo=df_multi.swaplevel(axis=1).stack()
    onsets_corr=stackedinfo.OnsetTime-stackedinfo.OnsetTime.values[0]
    RTonsets_corr=stackedinfo.RTTime-stackedinfo.OnsetTime.values[0]
    stackedinfo['CorrectedOnsetTime']=onsets_corr
    stackedinfo['CorrectedRTTime']=RTonsets_corr
    volume_timing=[720*i for i in range(0,406)]
    stackedinfo['VolumeAssignment']=np.digitize(onsets_corr.astype('int32'),volume_timing)
    stackedinfo['EventCol']=stackedinfo.index.to_frame(index=True)['Event'].values
    stackedinfo.EventCol[stackedinfo.EventCol == 'Stim']= stackedinfo[stackedinfo.EventCol == 'Stim'][['EventCol','BlockType','StimType']].apply(lambda x: '-'.join(x),axis=1)


    stackedinfo.to_csv(aef.replace('.txt','_filtered.csv'))






fs=glob.glob(os.path.join(eprimePath,'*/*filt*'))
df_list=[pd.read_csv(f,index_col=0) for f in fs]
sublist=['sub'+f.split('/')[0] for f in fs]
aggeventcols=pd.concat([df.EventCol for df in df_list],axis=1) 
aggeventcols.columns=sublist 
aggeventcols.apply(Counter,axis=1)
compareres=pd.concat([aggeventcols.apply(lambda x: max(Counter(x).keys(),key=operator.itemgetter(1)),axis=1),aggeventcols.apply(lambda x: Counter(x),axis=1)],axis=1)
mostcommon=aggeventcols.apply(lambda x: max(Counter(x).keys(),key=operator.itemgetter(1)),axis=1)
subswithsamedata=[k for k in aggeventcols.columns if np.sum(aggeventcols[k]==mostcommon) == 176]




flist=glob.glob(os.path.join(eprimePath,'*/*_3T_WM_run*_TAB_filtered.csv'))
dflist=[pd.read_csv(f,index_col=0) for f in flist]

accarr=np.zeros([405,865])
accarr[:,:]=np.nan
for i,df in enumerate(dflist):
    accarr[df.VolumeAssignment.values-1,i]=df.ACC.values
RTarr=np.zeros([405,865])
RTarr[:,:]=np.nan
for i,df in enumerate(dflist):
    RTarr[df.VolumeAssignment.values-1,i]=df.RT.values

sublist=list(map(lambda x: 'sub'+x.split('/')[1],flist)) 

#RTarr[np.isnan(RTarr)]=0
RTdf=pd.DataFrame(RTarr)
RTdf=RTdf.replace(to_replace=np.nan,method='ffill',limit=3)
RTdf.columns=sublist 
RTdf.to_csv(os.path.join(eprimePath,'ResponseTimeByTP.csv'))

#accarr[np.isnan(accarr)]=0
accdf=pd.DataFrame(accarr)
accdf=accdf.replace(to_replace=np.nan,method='ffill',limit=3)
accdf.columns=sublist 
accdf.to_csv(os.path.join(eprimePath,'AccuracyByTP.csv'))



RTdf_ffill=RTdf.replace(to_replace=np.nan,method='ffill',limit=3)
problem_tps=RTdf_ffill.ix[np.sum(RTdf_ffill == 0,axis=1) > 50]
subsindskeeps=problem_tps[~(problem_tps == 0)].dropna(axis=1).columns
substokeep=[sublist[s] for s in subsindskeeps]
np.save(os.path.join(eprimePath,'subswithlesszeros.npy'),substokeep)

eventcolarr=np.zeros([405,865],'object_')
for i,df in enumerate(df_list):
    eventcolarr[df.VolumeAssignment.values-1,i]=df.EventCol.values
ECdf=pd.DataFrame(eventcolarr)
ECdf.columns=sublist
ECdf.to_csv(os.path.join(eprimePath,'EventColbySub.csv'))