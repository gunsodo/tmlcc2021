import pymatgen

# Original source code: octadist
def number_to_symbol(x):
    """
    Convert atomic number to symbol and vice versa for atom 1-109.
    Parameters
    ----------
    x : str or int
        symbol or atomic number.
    Returns
    -------
    atom[x] : str
        If x is atomic number, return symbol.
    atom.index(i) : int
        If x is symbol, return atomic number.
    Examples
    --------
    >>> check_atom('He')
    2
    >>> check_atom(2)
    'He'
    """
    atoms = [
        '0',
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',
        'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
        'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
        'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
        'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
        'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
        'Db', 'Sg', 'Bh', 'Hs', 'Mt'
    ]

    if isinstance(x, int):
        return atoms[x]
    else:
        for i in atoms:
            if x == i:
                return atoms.index(i)

def get_coord_cif(f):
    """
    Get coordinate from .cif file.
    Parameters
    ----------
    f : str
        User input filename.
    Returns
    -------
    atom : list
        Full atomic labels of complex.
    coord : array_like
        Full atomic coordinates of complex.
    Examples
    --------
    >>> file = "example.cif"
    >>> atom, coord = get_coord_cif(file)
    >>> atom
    ['Fe', 'O', 'O', 'N', 'N', 'N', 'N']
    >>> coord
    array([[18.268051, 11.28912 ,  2.565804],
           [19.823874, 10.436314,  1.381569],
           [19.074466,  9.706294,  3.743576],
           [17.364238, 10.733354,  0.657318],
           [16.149538, 11.306661,  2.913619],
           [18.599941, 12.116308,  4.528988],
           [18.364987, 13.407634,  2.249608]])
    """
    import warnings
    warnings.filterwarnings('ignore')

    # works only with pymatgen <= v2021.3.3
    structure = pymatgen.Structure.from_file(f)
    atom = list(map(lambda x: number_to_symbol(x), structure.atomic_numbers))
    coord = structure.cart_coords

    return atom, coord