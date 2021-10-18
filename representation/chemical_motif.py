from cifAdapter import CIF2PandasAdapter
import pandas as pd
import numpy as np
import networkx as nx
import re
import os

src = 'mof_cif_pretest' # source folder
dst = 'pretest.csv' # csv destination

# calculate motif
motif_dict = {'Br': {0:('C')},
              'C': {1:('F','F','F','C'), 2:('H','H','H','C'), 3:('N','C'), 4:('C','C','C'), 5:('C','O','O'), 18:('I','C','C','C'), 20:('C','C','C','C'), 21:},
              'F': {6:('C')},
              'H': {7:('C')},
              'N': {8:('H','H','C'), 9:('O','O','C'), 10:{'C','C'}},
              'O': {11:('H','C')},
              'S': {12:('H','C')},
              'Cu': 13,
              'Cr': 14,
              'Zn': 15,
              'Zr': 16,
              'V': 17,
              'I': {19:('C')},
            }
def nn(string):
    return re.sub('[^a-zA-Z]','', string)
def nl(string):
    return re.sub('[^0-9]','', string)

def motif(G, motif_dict, volume):
    """
    G is a networkx graph with nodes as atoms and edges as bonds, 
    motif_dict is the above dictionary, 
    and lengths is a vector of the unit cell lengths, ### changed to volume
    i.e. [a,b,c], in Å
    """
    motif_key = []
    for a in motif_dict:
        mdict = motif_dict[a]
        if type(mdict) is dict:
            for m in mdict:
                motif_key.append(a + str(m))
        else:
            motif_key.append(a + str(mdict))
    count_dict = dict((i,0) for i in motif_key)
    for n in G.nodes(): # G here
        # remove the numerical index form the node name to get the element symbol
        elem = nn(n)
        # the neighbors should be sorted
        nbors = tuple(sorted(map(nn,G.neighbors(n)))) # G here
        mdict = motif_dict[elem]
        
        # we consider the metals separately (not as dictionaries)
        if type(mdict) is dict:
            for m in mdict:
                key = elem + str(m)
                pattern = tuple(sorted(mdict[m]))
                if pattern == nbors:
                    count_dict[key] += 1
                    # F should not be counted for the -CF3 motif
                    if m == 1:
                        count_dict['F6'] -= 3
        # for metals we do not compare neighbors
        else:
            count_dict[elem + str(mdict)] += 1
            
    # volume = lengths[0] * lengths[1] * lengths[2]
    for key in motif_key:
        count_dict[key] = float(count_dict[key])/volume
    
    number_density = np.array([[key, count_dict[key]] for key in motif_key])
    return np.array(sorted(number_density, key=lambda x: x[0]))

if __name__ == "__main__":
    df = []
    for filename in os.listdir(src):
        print(filename)
        adapter = CIF2PandasAdapter()
        output = adapter.apply(src + '/' + filename)
        G = nx.Graph()
        atoms = list(output.loops[0]["_atom_site_label"])
        site1 = list(output.loops[1]['_geom_bond_atom_site_label_1'])
        site2 = list(output.loops[1]['_geom_bond_atom_site_label_2'])
        for atom in atoms: # add atoms as node
            G.add_node(atom)
        for i in range(len(site1)): # add bonds
            G.add_edge(site1[i], site2[i])
        # use cell volume as volume
        vol = float(output.metadata.set_index(0).loc['_cell_volume'])
        desc = motif(G, motif_dict, vol)
        # save to df
        if os.path.exists(dst):
            df.loc[len(df.index)] = desc[:, 1]
        else:
            df = pd.DataFrame(columns=desc[:, 0])
            df.to_csv(dst)
            df.loc[len(df.index)] = desc[:, 1]
        G.clear()
    df.to_csv(dst, index=False)