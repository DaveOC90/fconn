import boto3
import botocore
import os
import pandas as pd
import sys
import pdb


sublistPath = sys.argv[1]
opDir = sys.argv[2]
motionEprimeOpt = int(sys.argv[3]) # 0 is motion 1 is Eprime

pmat_df=pd.read_csv(sublistPath) 
subs=list(map(str,pmat_df.Subject.values))

boto3.setup_default_session(profile_name='hcp')
s3 = boto3.resource('s3')
bucket = s3.Bucket('hcp-openaccess')

for sub in subs:
    prefLists = []

    if motionEprimeOpt == 1:
        #prefLists.append(['/'.join(['HCP_900',sub,'unprocessed/3T/tfMRI_WM_LR/LINKED_DATA/EPRIME/']),'.txt'])
        prefLists.append(['/'.join(['HCP_900',sub,'unprocessed/3T/tfMRI_EMOTION_LR/LINKED_DATA/EPRIME/']),'.txt'])
    elif motionEprimeOpt == 0:
        prefLists.append(['/'.join(['HCP_1200',sub,'MNINonLinear/Results/tfMRI_WM_LR/']),'Movement'])
        prefLists.append(['/'.join(['HCP_1200',sub,'MNINonLinear/Results/tfMRI_EMOTION_LR/']),'Movement'])
        prefLists.append(['/'.join(['HCP_1200',sub,'MNINonLinear/Results/rfMRI_REST1_LR/']),'Movement'])
    else:
        raise Exception('Third command line input must be 0/1 0 is motion 1 is Eprime')

    for prefList in prefLists:
        pL, filt = prefList
        suf = pL.split('/')[-2]
        destfold='/'.join([opDir,sub,suf.lower()])
        #print(destfold)
        if os.path.isdir(destfold):
            print(destfold, ' already exists, skipping download...')

        else:
            for obj in bucket.objects.filter(Prefix=pL):

                if all([x in obj.key for x in [filt]]):#['Movement']]):

                    destPathEnd = obj.key.replace(pL,'')
                    destPath = os.path.join(destfold,destPathEnd)

                    opfold = '/'.join(destPath.split('/')[:-1])

                    if not os.path.isdir(opfold):
                        print('Creating folder: ', opfold)
                        os.makedirs(opfold)

                    if os.path.isfile(destPath):
                        print('Already exists:',obj.key,'as:',destPath) 
                    else:
                        print('Downloading:',obj.key,'to:',destPath)
                        bucket.download_file(obj.key,destPath)
