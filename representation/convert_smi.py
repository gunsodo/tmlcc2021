from openbabel import pybel
import os
import pandas as pd
import re

counter=0
# initialize dataframe
df = pd.DataFrame({'MOFname':[], 'smi':[]})
# src
src = 'mof_cif_train'
# iterate through cifs
for filename in os.listdir(src):
    # print(filename)
    # check filename
    no = int(re.findall(r'\d+', filename)[0])
    mol = next(pybel.readfile("cif", src + "/" + filename))
    smi = mol.write('smi')
    sp = smi.split('\t')
    df.loc[len(df.index)] = [sp[1], sp[0]]
    counter += 1
    if counter % 200 == 0:
        df.to_csv('train.csv', index=False)

df.to_csv('train.csv', index=False)