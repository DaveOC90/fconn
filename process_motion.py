import pandas as pd
import glob
import sys
import os
import pdb


# Read in motion text files downloaded from HCP bucket, and create aggregate csv files

# Path to where the download is, with the folder structure based on hcpDownloadMotionEprime.py
motDir = sys.argv[1]


# Gather rest relative motion files and create csv
fs=sorted(glob.glob(os.path.join(motDir,'*/*rest*lr/*RelativeRMS.txt')))
dfs=[pd.read_csv(f,names=[f.split('/')[-3]]) for f in fs]
relMotDf=pd.concat(dfs,axis=1)
relMotDf.to_csv(os.path.join(motDir,'HCPRest1LRRelMotionParameters.csv'))


# Create rest motion csvs by timepoint so they can be used as confounds in dFC analysis
for i in range(1,35,2):
    meanDf = relMotDf.rolling(i,center=True).mean()

    meanDf.to_csv(os.path.join(motDir,'HCPRest1LRRelMotMeanCenterWin'+str(i).zfill(2)+'.csv'))


# Gather wm relative motion files and create csv
fs=sorted(glob.glob(os.path.join(motDir,'*/*wm*lr/*RelativeRMS.txt')))
dfs=[pd.read_csv(f,names=[f.split('/')[-3]]) for f in fs]
relMotDf=pd.concat(dfs,axis=1)
relMotDf.to_csv(os.path.join(motDir,'HCPWMLRRelMotionParameters.csv'))


# Create wm motion csvs by timepoint so they can be used as confounds in dFC analysis
for i in range(1,35,2):
    meanDf = relMotDf.rolling(i,center=True).mean()

    meanDf.to_csv(os.path.join(motDir,'HCPWMLRRelMotMeanCenterWin'+str(i).zfill(2)+'.csv'))
