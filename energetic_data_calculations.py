# Computing 2D convex hulls for binary data (M-H systems)
from __future__ import division
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
import xlrd

def extract_corrected_enthalpies(filename="./energetic_data/Enthalpies-correctionsZPE-sonia.xlsx"):
# 'filename' is the .xlsx file containing corrected energetic data
# The paths are absolute (for the sake of rapidity)
# which means the user needs to make sure the file
# is in the right directory.
    xl_workbook = xlrd.open_workbook(filename)
    sheet_names = xl_workbook.sheet_names()
    energetic_data = dict.fromkeys(sheet_names)

    # Extract H/M (column 0) and corrected DE (Delta E) (column 5) columns from all sheets
    # and return, for each sheet, a 2D array with H/M as first column and DE as second column.
    # Note: DE information is not available for the sheet 'Be'.
    for name in sheet_names:
        if name != 'Be':
            xl_sheet = xl_workbook.sheet_by_name(name)
            energetic_data[name] = np.column_stack((xl_sheet.col_values(0, 4), xl_sheet.col_values(9, 4)))
    return energetic_data

def construct_2D_convex_hull(e_data):
    '''
    Returns the reduced 2D convex hull of the 2D array 'e_data'.
    'e_data' contains the H/M proportions for a given H-M system (1st column)
    and the corrected enthalpies (2nd column).
    By 'reduced', we refer to the lower envelope of the convex hull.
    
    This method is particularly useful for computing the distance of a given
    binary hydride from the ground state.
    
    TODO: Raises error with 'Li' and 'Cr'. To be addressed.
    '''
    # First, the data set is augmented with the point (0, 0)
    data = np.concatenate((np.array([[0., 0.]]), e_data), axis=0)
    # Compute convex hull
    ch = ConvexHull(data)    
    # Get indices of points with minimum H/M ratio
    idx_min_hm = [i for i in ch.vertices if data[i][0] == min(data[:, 0])]
    # Get indices of points with maximum H/M ratio
    idx_max_hm = [i for i in ch.vertices if data[i][0] == max(data[:, 0])]
    # Select index of point with minimum H/M and minimum enthalpy
    ch_data = data[idx_min_hm, :]
    inf_idx = np.where(ch.vertices == idx_min_hm[np.argmin(ch_data[:, 1])])[0][0]
    # Select index of point with maximum H/M and minimum enthalpy
    ch_data = data[idx_max_hm, :]
    sup_idx = np.where(ch.vertices == idx_max_hm[np.argmin(ch_data[:, 1])])[0][0]
    # Return reduced convex hull (i.e. ground state)
    # We are interested only in the lower envelope of the convex hull
    #
    # Note: When plotting the ground state, the data should be augmented
    # with the point (0, 0)
    reduced_ch = []
    i = inf_idx
    reduced_ch.append(ch.vertices[inf_idx])
    while i != sup_idx:
        i = (i + 1) % len(ch.vertices)
        reduced_ch.append(ch.vertices[i])
    return np.array(reduced_ch) # ch.vertices[inf_idx:sup_idx + 1]

def project_on_ground_state_2D(x, aug_energetic_data, ground_state_idx):
    '''
    This function returns the 1D linear interpolants
    of the H/M data given in x to the ground state defined
    from aug_energetic_data (for augmented data with (0, 0)) and ground_state_idx.
    '''
    y = np.interp(x, aug_energetic_data[ground_state_idx, 0], aug_energetic_data[ground_state_idx, 1])
    return y

def enthalpies_pearsons_data(hm_dir='./energetic_data/HM_cif/'):
    '''
    H/M ratio files for Pearson's data are in .txt files
    in '../Asmas-data-and-scripts/energetic_data/HM_cif/'.
    
    This function returns a dictionary such that each key corresponds
    to a H-M system. For systems where energetic data have been computed
    with DFT calculations, the corresponding entry contains a 2D array
    with the first column containing H/M data, the second one containing
    enthalpies obtained via linear interpolation, and a third column containing 1 indicating that the compound is stable
    (we use 0 for unstable compounds).
    For systems where such information is not available, each entry of the dictionary contains a 1D
    array of H/M data only.
    '''
    # Extract H/M ratios of Pearson's data
    file_names = os.listdir(hm_dir)
    hm_files = [name for name in file_names if name[-3:] == 'txt']
    atomic_symbols = [name[:-4] for name in hm_files]
    dict_hm_ratios = dict.fromkeys(atomic_symbols)
    for name in hm_files:
        hm_ratios = np.loadtxt(hm_dir + name, ndmin=1)
        dict_hm_ratios[name[:-4]] = hm_ratios
    
    # For Pearson's H-M systems that exist in Natacha's data,
    # compute the convex hull of that system then project Pearson's
    # H/M ratios to deduce (by linear interpolation) enthalpies
    # for Pearson's hydrides
    e_data = extract_corrected_enthalpies()
    for key in e_data.keys():
        if key in atomic_symbols and key != 'Li' and key != 'Cr' and key != 'Be':
            red_ch = construct_2D_convex_hull(e_data[key])
            aug_data = np.concatenate((np.array([[0., 0.]]), e_data[key]), axis=0)
            y = project_on_ground_state_2D(dict_hm_ratios[key], aug_data, red_ch)
            dict_hm_ratios[key] = np.column_stack((dict_hm_ratios[key], y, np.zeros(len(y)), np.ones(len(y))))
    return dict_hm_ratios


def get_energetic_data():
    '''
    Merges energetic data from DFT calculations with Pearson's energetic data
    of the same H-M systems (i.e. some Pearson's systems that do not have
    a DFT countrepart are not merged).

    The result is a dictionary where each entry contains a 2D array with H/M
    data as first column, corrected enthalpies as second column, distances to ground state
    (differene between effective enthalpy and projection on ground state), and 0/1 if the
    compound is stable/unstable.

    IMPORTANT: Pearson's data are appended at the end of Natacha's data.
    It is important that POSCAR data and energetic data respect the same
    order (i.e. to associate the correct energetic data to each POSCAR).
    '''
    e_data = extract_corrected_enthalpies()
    e_data_pearson = enthalpies_pearsons_data()
    for key in e_data.keys():
        if key != 'Li' and key != 'Cr' and key != 'Be':
            red_ch = construct_2D_convex_hull(e_data[key])
            aug_data = np.concatenate((np.array([[0., 0.]]), e_data[key]), axis=0)
            dy = e_data[key][:, 1] - project_on_ground_state_2D(e_data[key][:, 0], aug_data, red_ch)
            is_stable = np.array([1 if x <= 0 else 0 for x in dy])
            e_data[key] = np.column_stack((e_data[key][:, 0], e_data[key][:, 1], dy, is_stable))
            if key in e_data_pearson.keys():
                e_data[key] = np.concatenate((e_data[key], e_data_pearson[key]))
    return e_data

# Example on how to plot the ground state using construct_2D_convex_hull(...)
# Uncomment to run
"""
key = 'Pd'
data = extract_corrected_enthalpies()[key]
red_ch = construct_2D_convex_hull(data)
aug_data = np.concatenate((np.array([[0., 0.]]), data), axis=0)
plt.scatter(aug_data[:, 0], aug_data[:, 1])
plt.grid()
plt.plot(aug_data[red_ch, 0], aug_data[red_ch, 1], marker='o', color='red')
f_size = 13
plt.title("Enthalpies of formation of binary metal hydrides", fontsize=f_size)
plt.xlabel(r"H$/M$", fontsize=f_size)
plt.ylabel(r"$\Delta H$", fontsize=f_size)
plt.show()
# And on how to project Pearson's H/M data on the ground state using enthalpies_pearsons_data()
x, y = enthalpies_pearsons_data()[key][:, 0], enthalpies_pearsons_data()[key][:, 1]
plt.scatter(x, y, marker='x', c='orange')
"""