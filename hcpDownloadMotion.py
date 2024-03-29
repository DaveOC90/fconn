import boto3
import botocore
import os
import pandas as pd
import sys
import pdb

# credspath='hcp_aws_keys.txt'
# bucket_name='hcp-openaccess'

# creds=[s.strip().split('=')[1] for s in open(credspath,'r')]

# aws_access_key_id,aws_secret_access_key=creds
# session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
#                                 aws_secret_access_key=aws_secret_access_key)
# s3_resource = session.resource('s3', use_ssl=True)

# bucket = s3_resource.Bucket(bucket_name)

# s3_resource.meta.client.head_bucket(Bucket=bucket_name)

sublistPath = sys.argv[1]
opDir = sys.argv[2]

pmat_df=pd.read_csv(sublistPath) 
subs=list(map(str,pmat_df.Subject.values))

boto3.setup_default_session(profile_name='hcp')
s3 = boto3.resource('s3')
bucket = s3.Bucket('hcp-openaccess')

for sub in subs:
    prefList = []
    prefList.append('/'.join(['HCP_900',sub,'unprocessed/3T/tfMRI_WM_LR/LINKED_DATA/EPRIME/']))
    #prefList.append('/'.join(['HCP_1200',sub,'MNINonLinear/Results/tfMRI_WM_LR/']))
    #prefList.append('/'.join(['HCP_1200',sub,'MNINonLinear/Results/rfMRI_REST1_LR/']))


    for pL in prefList:
        suf = pL.split('/')[-2]
        destfold='/'.join([opDir,sub,suf.lower()])
        print(destfold)

        if os.path.isdir(destfold):

            pass

        else:

            for obj in bucket.objects.filter(Prefix=pL):

                if all([x in obj.key for x in ['.txt']]):#['Movement']]):

                    destPathEnd = obj.key.replace(pL,'')
                    destPath = os.path.join(destfold,destPathEnd)

                    opfold = '/'.join(destPath.split('/')[:-1])

                    try:
                        os.makedirs(opfold)
                    except FileExistsError:
                        pass

                    if os.path.isfile(destPath):
                        print('Already exits:',obj.key,'as:',destPath) 
                    else:
                        print('Downloading:',obj.key,'to:',destPath)
                        bucket.download_file(obj.key,destPath)
