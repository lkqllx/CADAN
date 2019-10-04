from __future__ import print_function

import numpy as np
import scipy.io as sio


def get_filenames1(datadir='data', n_gauss=512, n_fac=300):
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
    basename = '_plda_checked_t' + str(n_fac) + '_w_' + str(n_gauss) + 'c-'
    for i in range(1, 5):
        trn_matfile.append(datadir + 'male' + basename + str(i) + '.mat')
        trn_gender.append('male')
    for i in range(1, 5):
        trn_matfile.append(datadir + 'female' + basename + str(i) + '.mat')
        trn_gender.append('female')

    return trn_matfile, trn_gender


def get_filenames2(datadir='data', n_gauss=1024, n_fac=500, str = 'mic'):
    """
    # Get the filenames of i-vector files and their corresponding gender
    # :param datadir: Directory containing data
    # :param n_gauss: No. of Gaussians in the UBM
    # :param n_fac: No. of speaker factors (dim of i-vectors)
    # :return: List of .mat file and their gender
    """
    # male_target-tel_mix_t500_w_1024c
    matfile = []
    gender = []
    channel = []
    # matfile.append(datadir + 'male_target-' + 'mic' + '_mix_t500_w_1024c.mat')
    # gender.append('male')
    # channel.append('mic')
    # matfile.append(datadir + 'female_target-' + 'mic' +'_mix_t500_w_1024c.mat')
    # gender.append('female')
    # channel.append('mic')
    # matfile.append(datadir + 'male_target-' + 'tel' + '_mix_t500_w_1024c.mat')
    # gender.append('male')
    # channel.append('tel')
    # matfile.append(datadir + 'female_target-' + 'tel' +'_mix_t500_w_1024c.mat')
    # gender.append('female')
    # channel.append('tel')
    matfile.append(datadir + 'male_target-' + 'mic-' + '06dB' + '_mix_t500_w_1024c.mat')
    gender.append('male')
    channel.append('mic')
    matfile.append(datadir + 'female_target-' + 'mic-' + '06dB'+'_mix_t500_w_1024c.mat')
    gender.append('female')
    channel.append('mic')
    matfile.append(datadir + 'male_target-' + 'tel-' + '06dB' '_mix_t500_w_1024c.mat')
    gender.append('male')
    channel.append('tel')
    matfile.append(datadir + 'female_target-' + 'tel-' + '06dB' +'_mix_t500_w_1024c.mat')
    gender.append('female')
    channel.append('tel')
    # matfile.append(datadir + 'male_target-' + 'mic-' + '15dB' + '_mix_t500_w_1024c.mat')
    # gender.append('male')
    # channel.append('mic')
    # matfile.append(datadir + 'female_target-' + 'mic-' + '15dB'+'_mix_t500_w_1024c.mat')
    # gender.append('female')
    # channel.append('mic')
    # matfile.append(datadir + 'male_target-' + 'tel-' + '15dB' '_mix_t500_w_1024c.mat')
    # gender.append('male')
    # channel.append('tel')
    # matfile.append(datadir + 'female_target-' + 'tel-' + '15dB' +'_mix_t500_w_1024c.mat')
    # gender.append('female')
    # channel.append('tel')
    # basename = '_plda_checked_t' + str(n_fac) + '_w_' + str(n_gauss) + 'c-'
    # for i in range(1, 5):
    #     matfile.append(datadir + 'male' + basename + str(i) + '.mat')
    #     gender.append('male')
    # for i in range(1, 5
    #                ):
    #     matfile.append(datadir + 'female' + basename + str(i) + '.mat')
    #     gender.append('female')

    return matfile, gender, channel


# Load data from .mat files. Return the i-vectors, gender labels
# (0 for male, 1 for female) and speaker labels (0..nSpks-1)
def load_ivectors(mat_files, gender, channel, shuffle=False):
    data = sio.loadmat(mat_files[0])
    dim = data['w'].shape[1]
    ivec = np.empty((0, dim), dtype='float32')
    gender_lbs = np.empty(0)
    channel_lbs = np.empty(0)
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

        if channel[i] == 'tel':
            l_channel = np.zeros(n_vecs, dtype='int32')
        else:
            l_channel = np.ones(n_vecs, dtype='int32')

        ivec = np.vstack((ivec, w))
        gender_lbs = np.hstack((gender_lbs, l))
        channel_lbs = np.hstack((channel_lbs, l_channel))
        spk_logical = data['spk_logical']
        for spk in spk_logical:
            id_str = spk.tolist()[0].tolist()[0]
            id_str = gender[i] + '-' + id_str
            spk_ids.append(id_str)
    # Assign speaker labels to i-vectors
    _, spk_lbs = np.unique(spk_ids, return_inverse=True)

    # Shuffle the order of the vectors
    # if shuffle:
    #     idx = np.random.permutation(ivec.shape[0])
    #     ivec = ivec[idx, :]
    #     gender_lbs = gender_lbs[idx]
    #     spk_lbs = spk_lbs[idx]

    return ivec, gender_lbs, spk_lbs, channel_lbs
# def load_ivectors(mat_files, gender, shuffle=False):
#     data = sio.loadmat(mat_files[0])
#     dim = data['w'].shape[1]
#     ivec = np.empty((0, dim), dtype='float32')
#     gender_lbs = np.empty(0)
#     spk_ids = list()
#     n_files = len(mat_files)
#     for i in range(0, n_files):
#         print('Loading ' + mat_files[i])
#         data = sio.loadmat(mat_files[i])
#         w = np.array(data['w'], dtype='float32')
#         n_vecs = w.shape[0]
#         if gender[i] == 'male':
#             l = np.zeros(n_vecs, dtype='int32')
#         else:
#             l = np.ones(n_vecs, dtype='int32')
#         ivec = np.vstack((ivec, w))
#         gender_lbs = np.hstack((gender_lbs, l))
#         spk_logical = data['spk_logical']
#         for spk in spk_logical:
#             id_str = spk.tolist()[0].tolist()[0]
#             id_str = gender[i] + '-' + id_str
#             spk_ids.append(id_str)
#
#     # Assign speaker labels to i-vectors
#     _, spk_lbs = np.unique(spk_ids, return_inverse=True)
#
#     # Shuffle the order of the vectors
#     if shuffle:
#         idx = np.random.permutation(ivec.shape[0])
#         ivec = ivec[idx, :]
#         gender_lbs = gender_lbs[idx]
#         spk_lbs = spk_lbs[idx]
#
#     return ivec, gender_lbs, spk_lbs

def extract_data(x, spk_lbs, min_n_vecs, n_spks=-1, c_lbs=-1, shuffle=False):
    idx_lst = list()
    new_spk_lbs = list()
    out_idx_lst = list()
    out_spk_lbs = list()
    unq_spk_lbs = np.unique(np.asarray(spk_lbs))        # Find a list of unique speaker labels
    count = 0
    out_count = 0# New spk labels start from 0
    if n_spks == -1:
        n_spks = len(unq_spk_lbs)                       # Default is to extract the data of all speakers
    for lbs in unq_spk_lbs:
        idx = [t for t, e in enumerate(spk_lbs) if e == lbs]  # Find idx in spk_trn that match spkid
        if len(idx) >= min_n_vecs:
            channel_lbs = c_lbs[idx]
            tel_idx = [t for t, e in enumerate(channel_lbs) if e == 0]
            mic_idx = [t for t, e in enumerate(channel_lbs) if e == 1]
            idx_arrray = np.asarray(idx)
            if len(tel_idx)>10 and len(mic_idx)>25:
                idx_lst.extend(idx_arrray[mic_idx])
                new_spk_lbs.extend([count] * len(mic_idx))
                out_idx_lst.extend(idx_arrray[tel_idx])
                out_spk_lbs.extend([count] * len(tel_idx))
                # Repeat count len(idx) times. Assign new labels
                count = count + 1
        if count == n_spks:
            break
        #
        # elif len(idx) < 3*min_n_vecs/4 and len(idx) > min_n_vecs/2:
        #     out_idx_lst.extend(idx)
        #     out_spk_lbs.extend([out_count] * len(idx))      # Repeat count len(idx) times. Assign new labels
        #     out_count = out_count + 1
        #     if out_count == n_spks:
        #         break

    new_x = x[idx_lst, :]
    out_x = x[out_idx_lst, :]
    # if shuffle:
    #
    #     new_spk_lbs = np.asarray(new_spk_lbs)[ridx]

    return new_x, new_spk_lbs, out_x, out_spk_lbs
# def extract_data(x, spk_lbs, min_n_vecs, n_spks=-1, shuffle=False):
#     idx_lst = list()
#     new_spk_lbs = list()
#     out_idx_lst = list()
#     out_spk_lbs = list()
#     unq_spk_lbs = np.unique(np.asarray(spk_lbs))        # Find a list of unique speaker labels
#     count = 0
#     out_count = 0# New spk labels start from 0
#     if n_spks == -1:
#         n_spks = len(unq_spk_lbs)                       # Default is to extract the data of all speakers
#     for lbs in unq_spk_lbs:
#         idx = [t for t, e in enumerate(spk_lbs) if e == lbs]  # Find idx in spk_trn that match spkid
#         if len(idx) > min_n_vecs:
#             idx_lst.extend(idx)
#             new_spk_lbs.extend([count] * len(idx))
#             count = count + 1
#             if count == n_spks:
#                 break
#         elif len(idx) < 3 * min_n_vecs / 4 and len(idx) > min_n_vecs / 2:
#             out_idx_lst.extend(idx)
#             out_spk_lbs.extend([out_count] * len(idx))      # Repeat count len(idx) times. Assign new labels
#             out_count = out_count + 1
#             if out_count == n_spks:
#                 break
#         #
#         # elif len(idx) < 3*min_n_vecs/4 and len(idx) > min_n_vecs/2:
#         #     out_idx_lst.extend(idx)
#         #     out_spk_lbs.extend([out_count] * len(idx))      # Repeat count len(idx) times. Assign new labels
#         #     out_count = out_count + 1
#         #     if out_count == n_spks:
#         #         break
#
#     new_x = x[idx_lst, :]
#     out_x = x[out_idx_lst, :]
#     # if shuffle:
#     #
#     #     new_spk_lbs = np.asarray(new_spk_lbs)[ridx]
#
#     return new_x, new_spk_lbs, out_x, out_spk_lbs
