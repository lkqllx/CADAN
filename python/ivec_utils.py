from __future__ import print_function
from collections import Counter
import numpy as np
import scipy.io as sio


def get_filenames1(datadir='data/', n_gauss=512, n_fac=300, noise_types=['mic']):
    """
    # Get trn and tst filenames and their corresponding gender
    # Consider the first 3 i-vector files as training file and the 4th
    # i-vector file as the test file.
    # :param datadir: Directory containing data
    # :param n_gauss: No. of Gaussians in the UBM
    # :param n_fac: No. of speaker factors (dim of i-vectors)
    # :return: List of .mat files and their genders for train and test
    # """

    # Use ivecFile1 to ivecFile3 for training
    trn_matfile = []
    trn_gender = []
    for noise in noise_types:
        trn_matfile.append(f'{datadir}male_target-{noise}-06dB_t{n_fac}_w_{n_gauss}c.mat')
        trn_gender.append('male')
    for noise in noise_types:
        trn_matfile.append(f'{datadir}female_target-{noise}-06dB_t{n_fac}_w_{n_gauss}c.mat')
        trn_gender.append('female')
    return trn_matfile, trn_gender


# Load data from .mat files. Return the i-vectors, gender labels
# (0 for male, 1 for female) and speaker labels (0..nSpks-1)
def load_ivectors(mat_files, gender, shuffle=False):
    data = sio.loadmat(mat_files[0])
    dim = data['w'].shape[1]
    ivec = np.empty((0, dim), dtype='float32')
    gender_lbs = np.empty(0)
    spk_ids = list()
    n_files = len(mat_files)
    for i in range(0, n_files):
        print('Loading ' + mat_files[i])
        data = sio.loadmat(mat_files[i])
        w = np.array(data['w'], dtype='float32')
        n_vecs = w.shape[0]
        if gender[i] == 'male':
            l = np.zeros(n_vecs, dtype='int32')
        else:
            l = np.ones(n_vecs, dtype='int32')

        ivec = np.vstack((ivec, w))
        gender_lbs = np.hstack((gender_lbs, l))
        spk_logical = data['spk_logical']
        for spk in spk_logical:
            id_str = spk.tolist()[0].tolist()[0]
            id_str = gender[i] + '-' + id_str
            spk_ids.append(id_str)
    # Assign speaker labels to i-vectors
    _, spk_lbs = np.unique(spk_ids, return_inverse=True)

    return ivec, gender_lbs, spk_lbs,


def extract_data(x: np.ndarray, spk_lbs, min_n_vecs, shuffle=False):
    valid_indexes = list()
    unq_spk_lbs = np.unique(np.asarray(spk_lbs))
    for lbs in unq_spk_lbs:
        idx = [t for t, e in enumerate(spk_lbs) if e == lbs]
        if len(idx) >= min_n_vecs:
            valid_indexes += idx
    valid_x = x[valid_indexes]
    valid_spk_lbs = spk_lbs[valid_indexes]
    return valid_x, valid_spk_lbs

