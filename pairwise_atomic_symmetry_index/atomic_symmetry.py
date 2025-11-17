import copy

import numpy as np

elements_table = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'TI': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fi': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}


def StrFromMol(ISI):
    """
    Extract atomic structure information

    Parameters
    ----------
    ISI : list of str
        structure information

    Returns
    -------
    si : ndarray
        Array of atomic black information
    n_atom : int
        Number of atom
    n_adj : int
        Number of bond
    """
    ISI = copy.deepcopy(ISI)
    si = []
    n_atom_start = 4
    n_atom = int(ISI[3][0:3])
    n_adj = int(ISI[3][3:6])
    for j in range(n_atom_start, n_atom_start + n_atom):
        si.append(ISI[j].split())
    si = np.array(si)

    return si, n_atom, n_adj


def StepBondFromMol(n_atom, n_adj, ISI):
    """
    Compute adjacency matrix (Sa) and bond  matrix (Sbon)

    Parameters
    ----------
    n_atom : int
        Number of atom
    n_adj : int
        Number of bond
    ISI : list of str
        structure information

    Returns
    -------
    Sa : ndarray
        Adjacency matrix
    Sbon : ndarray
        bond matrix
    """
    n_atom_start = 4
    ISI = copy.deepcopy(ISI)
    Sa = np.zeros((n_atom, n_atom))
    Sbon = np.zeros((n_atom, n_atom))
    for j in range(n_atom_start + n_atom, n_atom_start + n_atom + n_adj):
        i_s = min([int(ISI[j][0:3]) - 1, int(ISI[j][3:6]) - 1])
        i_g = max([int(ISI[j][0:3]) - 1, int(ISI[j][3:6]) - 1])
        Sa[i_s, i_g] = 1
        Sbon[i_s, i_g] = int(ISI[j][6:9])
    Sa = Sa + Sa.T
    Sbon = Sbon + Sbon.T

    return Sa, Sbon


def adjacent_info(n_atom, Latom, Sa):
    """
    Generate adjacency lists for each atom in a molecule.

    Parameters
    ----------
    n_atom : int
        Number of atom
    Latom : list
        List of atom symbols (e.g., ['C', 'H', 'O', ...]).
    Sa : ndarray
        Adjacency matrix

    Returns
    -------
    Ladj : list of list of int
        List for each atom in a molecule (e.g., [[2], [1,3], [2,4,5], ...]).
    Ladj_H : list of list of int
        List of hydrogen atoms for each atom in a molecule (e.g., [[], [], [],[6,7,8], ...]).
    """
    Ladj_H = []
    Ladj = []
    for j in range(0, n_atom):
        Ladj_ = np.where(Sa[j, :] > 0)
        Ladj.append(list(1 + Ladj_[0]))
        Ladj__ = []
        Ladj_nH_ = []
        for i in Ladj_[0]:
            if Latom[i] == 'H':
                Ladj__.append(i + 1)
            else:
                Ladj_nH_.append(i + 1)
        Ladj_H.append(Ladj__)

    return Ladj, Ladj_H


def FullStep(n_atom, Sa, Ladj):
    """
    Compute the full-step matrix (topological distance matrix) for a molecule.

    Parameters
    ----------
    n_atom : int
        Number of atom
    Sa : ndarray
        Adjacency matrix
    Ladj : list of list of int
        List for each atom in a molecule (e.g., [[2], [1,3], [2,4,5], ...]).

    Returns
    -------
    SF : ndarray
        Full-step matrix
    """
    step_max = n_atom - 1
    SF = copy.deepcopy(Sa)
    w_ms_r, w_ms_c = np.where(SF == 1)
    for m in range(1, step_max):
        if len(w_ms_r) == 0:
            break
        w_ms_r_ = []
        w_ms_c_ = []
        for i in range(0, len(w_ms_r)):
            w_msi = Ladj[w_ms_c[i]]
            for j in w_msi:
                if SF[w_ms_r[i], j - 1] == 0 and w_ms_r[i] != j - 1:
                    SF[w_ms_r[i], j - 1] = m + 1
                    w_ms_r_.append(w_ms_r[i])
                    w_ms_c_.append(j - 1)
        w_ms_r = w_ms_r_
        w_ms_c = w_ms_c_

    return SF


def M_del(M, del_n):
    """
    Delete specified rows and columns from a matrix.

    Parameters
    ----------
    M : ndarray
        Input matrix.
    del_n : ndarray
        Index or list of indices of rows/columns to delete.

    Returns
    -------
    M_new : ndarray
        A new matrix with deleted rows and columns.
    """
    M_new = copy.deepcopy(M)
    M_new = np.delete(M_new, del_n, 0)
    M_new = np.delete(M_new, del_n, 1)

    return M_new


def get_mol_info(mol_file):
    """
    molecular structure information from a .mol file.

    Parameters
    ----------
    mol_file : str
        a .mol file

    Returns
    -------
    mol_info : dict
        Dictionary containing the following molecular information:
        - 'Latom' : list of str
            List of atomic symbols.
        - 'Sa' : ndarray
            Adjacency matrix.
        - 'Sbon' : ndarray
            bond matrix.
        - 'Ladj_H' : list of list of int
            List of hydrogen atoms for each atom in a molecule
        - 'SF' : ndarray
            Full-step (topological distance) matrix.
    """
    with open(mol_file, 'r') as re:
        mol_lines = re.readlines()
    si, n_atom, n_adj = StrFromMol(ISI=mol_lines)
    Latom = si[:, 3].tolist()
    Sa, Sbon = StepBondFromMol(n_atom=n_atom, n_adj=n_adj, ISI=mol_lines)
    Ladj, Ladj_H = adjacent_info(n_atom=n_atom, Latom=Latom, Sa=Sa)
    SF = FullStep(n_atom=n_atom, Sa=Sa, Ladj=Ladj)
    mol_info = {
        'Latom': Latom,
        'Sa': Sa,
        'Sbon': Sbon,
        'Ladj_H': Ladj_H,
        'SF': SF
    }

    return mol_info


def generate_pasi(mol_file):
    mol_info = get_mol_info(mol_file=mol_file)
    Latom = mol_info['Latom']
    Sa = mol_info['Sa']
    SF = mol_info['SF']
    Sbon = mol_info['Sbon']
    Ladj_H = mol_info['Ladj_H']
    notH = np.where(np.array(Latom) != 'H')[0]
    iH = np.where(np.array(Latom) == 'H')[0]
    Sa_notH = M_del(M=Sa, del_n=iH)
    SF_notH = M_del(M=SF, del_n=iH)
    Sbon_notH = M_del(M=Sbon, del_n=iH)
    degree_of_branch_notH = np.sum(Sa_notH, axis=1)
    bond_sum_notH = np.sum(Sbon_notH, axis=1)

    bond_multiply_notH = np.zeros((len(notH), 1))
    for i in range(len(notH)):
        bond_multiply_notH[i,] = np.prod(Sbon_notH[i, np.nonzero(Sbon_notH[i, :])], axis=1)
    atomic_number_notH = []
    adj_H_num = []
    for atom_i, atom in enumerate(Latom):
        if atom_i in notH:
            atomic_number_notH.append(elements_table[f'{atom}'])
            adj_H_num.append(len(Ladj_H[atom_i]))
    atom_type_info = np.c_[atomic_number_notH, degree_of_branch_notH, bond_sum_notH, bond_multiply_notH, adj_H_num]
    atom_type_info_dict = {}
    for i in range(len(notH)):
        atom_type_info_dict[
            f'{i}'] = f'{atom_type_info[i, 0]}|{atom_type_info[i, 1]}|{atom_type_info[i, 2]}|{atom_type_info[i, 3]}|{atom_type_info[i, 4]}'
    SF_notH_sort = np.sort(SF_notH, axis=0, kind='mergesort').astype(str)
    SF_notH_sort = np.char.add(SF_notH_sort, '|')
    SF_notH_sort_index = np.argsort(SF_notH, axis=0, kind='mergesort')
    atom_type_info_m = np.zeros((len(notH), len(notH)), dtype=object)
    for i in range(len(notH)):
        atom_type_info_m[SF_notH_sort_index == i] = atom_type_info_dict[f'{i}']
    atom_type_info_m = atom_type_info_m.astype(str)
    SF_atom_type_info_combination0 = np.char.add(SF_notH_sort, atom_type_info_m)
    SF_atom_type_info_combination = np.sort(SF_atom_type_info_combination0, axis=0, kind='mergesort')
    Mpasi = np.zeros((len(notH), len(notH)))
    for i in range(len(notH) - 1):
        for j in range(i + 1, len(notH)):
            ssw = np.where(SF_atom_type_info_combination[:, i] == SF_atom_type_info_combination[:, j])[0]
            Mpasi[i, j] = len(ssw) / len(notH)
            Mpasi[j, i] = len(ssw) / len(notH)

    return Mpasi


if __name__ == "__main__":
    mol_file = r'test.mol'
    Mpasi = generate_pasi(mol_file=mol_file)
    print(Mpasi)
