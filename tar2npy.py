import tarfile
import io
import pandas as pd
import sys, os
import glob
import pdb 
import numpy as np
pathTars = sys.argv[1]

# Figure out how stdout is encoded for this system
encodeFmt = sys.stdout.encoding.lower()


tars = glob.glob(os.path.join(pathTars,'*_REST_*.tar.gz'))


for t in sorted(tars):
    print(t)
    # Init tar object
    file_obj= tarfile.open(t,"r")

    # get all files in tar
    namelist=file_obj.getnames()

    namelistGSR = [nl for nl in namelist if '_GSR_' in nl]
    namelistNOGSR = [nl for nl in namelist if '_NOGSR_' in nl]

    dctGSR = {}
    dctNOGSR = {}

    print('Extracting GSR Data')
    for nlg in namelistGSR:
        subid = nlg.split('/')[-1].split('_')[0]

        # Extract encoded bit version of file content
        fileExtract = file_obj.extractfile(nlg)

        # Decode file contents and treat as "in-memory file-like object"
        fileSio = io.StringIO(fileExtract.read().decode(encodeFmt))

        # read csv
        df = pd.read_csv(fileSio, sep="\t",index_col=0)
        dctGSR[subid] = df.dropna(axis=1).values


    gsroppath = t.replace('.tar.gz','_GSR.npy')
    print('Saving to: ',gsroppath)
    np.save(gsroppath,dctGSR)

    print('Extracting No GSR Data')

    for nln in namelistNOGSR:
        subid = nln.split('/')[-1].split('_')[0]

        # Extract encoded bit version of file content
        fileExtract = file_obj.extractfile(nln)

        # Decode file contents and treat as "in-memory file-like object"
        fileSio = io.StringIO(fileExtract.read().decode(encodeFmt))

        # read csv
        df = pd.read_csv(fileSio, sep="\t",index_col=0)
        dctNOGSR[subid] = df.dropna(axis=1).values



    nogsroppath = t.replace('.tar.gz','_NOGSR.npy')
    print('Saving to: ',nogsroppath)
    np.save(nogsroppath,dctNOGSR)

