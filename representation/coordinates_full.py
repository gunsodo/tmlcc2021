from tmlcc2021.representation.coordinates_core import get_coord_cif
import numpy as np
import pandas as pd
import octadist as oc
import os
from ase import Atoms

file = pd.read_csv('/data/coordinates/train.csv')
#file = file.iloc[:,:]

mof_data = []
mof_valid = []
mof_invalid = []

for mof in range(0,file.shape[0]):
    try:
        elements = file.iloc[mof]['atoms']
        elements_arr = elements.strip("[]'").split("', '")
        
        lattice = Atoms(elements_arr)
        elements_num_arr = lattice.get_atomic_numbers()

        coords = file.iloc[mof]['coords']
        coords_arr = list(filter(lambda x: x!='', coords.replace('[', "").replace(']', "").replace('\n', "").strip(" ").split(' ')))

        mof_coords = np.array([int(elements_num_arr[0]),float(coords_arr[0]),float(coords_arr[1]),float(coords_arr[2])])
        for i in range(1,len(elements_arr)):
            mof_coords = np.vstack((mof_coords,[int(elements_num_arr[i]),float(coords_arr[3*i]),float(coords_arr[3*i+1]),float(coords_arr[3*i+2])]))
        
        mof_valid.append(mof)
        mof_data.append(mof_coords)
    except:
        #print(file.iloc[mof]['MOFname'],' failed!, No of atoms: ', num_atoms[mof])
        mof_invalid.append(mof)

from pymatgen.core import *

import zipfile as zf


    
new_mof_data = []

archive = zf.ZipFile('/mof_cif/mof_cif_test.zip', 'r')  #Access zip file

for mof_index in mof_invalid:
    test = archive.extract(str('mof_cif_test/' + file.iloc[mof_index]['MOFname'] + ".cif"))  # Read .cif file from zip

    atom_full, coord_full = get_coord_cif(test)
    atomic_numbers = Atoms(atom_full).get_atomic_numbers()
    
    new_mof_coords = np.array([float(atomic_numbers[0]),float(coord_full[0][0]),float(coord_full[0][1]),float(coord_full[0][2])])
    for i in range(1,len(atomic_numbers)):
        new_mof_coords = np.vstack((new_mof_coords,np.array([float(atomic_numbers[i]),float(coord_full[i][0]),float(coord_full[i][1]),float(coord_full[i][2])])))

    new_mof_data.append(new_mof_coords)
  
mof_data = mof_data + new_mof_data
mof_index_all = mof_valid + mof_invalid

num_atoms = np.zeros_like(mof_data)
for i in range(0,file.shape[0]):
    num_atoms[i] = len(mof_data[i])

max_atom = max(num_atoms)

mof_data_pad = np.zeros_like(mof_data)
for i in range(0,file.shape[0]):
    mof_data_pad[i] = np.pad(mof_data[i], ((0, int(max_atom - len(mof_data[i]))), (0, 0)), 'constant', constant_values=(0))
    mof_data_pad[i] = mof_data_pad[i][mof_data_pad[i][:, 0].argsort()]

mof_name_new = []
for i in range(0,file.shape[0]):
    mof_name_new.append(file.iloc[mof_index_all[i]]['MOFname'])

new_mof_df = pd.DataFrame(data=np.transpose([mof_name_new,num_atoms,mof_data_pad]), index=mof_index_all, columns=['MOFname','Num_atoms','Coords'])

new_mof_df = new_mof_df.sort_index()
new_mof_coords = np.array(new_mof_df.iloc[:]['Coords'])
new_mof_df = new_mof_df.drop(['Coords'], axis=1)

for i in range(0,max_atom):
    atom_column = []

    for j in range(0,new_mof_df.shape[0]):
        atom_column.append(new_mof_coords[j][max_atom-i-1][:])
    
    new_mof_df[str('atom_'+str(i+1))] = atom_column

new_mof_df.to_csv('/work/Export/train_coords_full.csv',index=False)


