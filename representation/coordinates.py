import octadist as oc
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

base_path = "your/path/mof_cif_pretest"

atoms = []
coords = []
filenames = []
errors = []

files = sorted(os.listdir(base_path))

for file in tqdm(files):
    full_path = os.path.join(base_path, file)
    try:
        atom_full, coord_full = oc.io.extract_coord(full_path)
        atoms.append(atom_full)
        coords.append(coord_full)
        filenames.append(os.path.splitext(file)[0])
    except:
        errors.append(os.path.splitext(file)[0])

df = pd.DataFrame({'MOFName': filenames, 'atoms': atoms, 'coords': coords})
df.to_csv("../results/output.csv", index=False)
print(errors)

