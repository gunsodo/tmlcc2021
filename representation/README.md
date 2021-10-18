# Representation
## cifAdapter
cifAdapter.py convert .cif files to pandas dataframe:
'''
from cifAdapter import CIF2PandasAdapter
mof = CIF2PandasAdapter().apply(filename) # pandas dataframe
'''

## Atomic Property-weighted Radial Distribution Function (AP-RDF)
add the desired property to atomic_property_dict.py. modify line 20-49 to parameters of choice in APRDF.py and run APRDF.py to generate a descriptor csv file.

## Bag-of-atoms (BOA)
edit directory on line 16 of gen-bag-of-atoms.py and bag-of-atoms.py. Running bag-of-atoms.py will generate a csv file containing the 216 epsilon and 216 sigma "bags" with their corresponding atoms. Then, run gen-bag-of-atoms to generate a descriptor csv file.
