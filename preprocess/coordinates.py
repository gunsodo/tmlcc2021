import octadist as oc
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

base_path = "/Users/gunsodo/Desktop/GunSoDo/Workspace/tmlcc/tmlcc-2021/mof_cif_train"

atoms = []
coords = []
filenames = []
errors = []

files = sorted(os.listdir(base_path))[60000:]

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
df.to_csv("../results/train_coordinates_7.csv", index=False)
print(errors)

# print(type(atom_full), type(coord_full))

# my_plot = oc.draw.DrawComplex_Matplotlib(atom=atom_full, coord=coord_full)
# my_plot.add_atom()
# my_plot.add_bond()
# my_plot.fig.suptitle("")
# my_plot.save_img(f"../results/pics/{file[:-4]}")

