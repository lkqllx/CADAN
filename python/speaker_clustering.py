"""
Perform speaker clustering by applying spectral clustering (SC) or agglomerative
hierarchical clustering (AHC) on a pairwise PLDA similarity matrix

M.W. Mak, Nov. 2017
"""

import sys
import h5py as h5
import numpy as np
from lib.h5helper import h52dict
from matplotlib.pyplot import imshow, bar
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import os.path
from lib.pairwise_scoring import get_plda_matrix, score2sim, save_scoremat


# Read a list of i-vector file (in .h5 format), perform speaker clustering and
# save the resulting i-vector file (in .h5 format) with spk_ids determined
# by the clustering algorithm. clus_method can be either 'sc' or 'ac'.
# Return spk_lbs, scr_mat, dist_mat, and sim_mat
def speaker_clustering(infiles, outfile, modelfile, scr_mat=None, partial=False,
                       prep_mode='lennorm', clus_method='sc', n_clusters=10, sigma=100):
    # Load i-vectors, ignoring the speaker IDs
    X, _, n_frames, spk_path = load_data(infiles)

    # Load PLDA model
    model = h52dict(modelfile)

    # Compute PLDA score matrix if not provided
    if scr_mat is None:
        scr_mat = get_plda_matrix(X, model, prep_mode, partial=partial)

    # Convert score matrix to similarity and distance matrix
    sim_mat, dist_mat = score2sim(scr_mat, sigma=sigma)

    # Perform clustering
    if clus_method == 'sc':
        spk_lbs = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                     affinity='precomputed').fit(sim_mat).labels_
    elif clus_method == 'ac':
        spk_lbs = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete',
                                          affinity='precomputed').fit(dist_mat).labels_

    # Append 'spk-' to the numerical labels and assign them to spk_ids
    unicode = h5.special_dtype(vlen=str)
    spk_ids = []
    for i in range(len(spk_lbs)):
        spk_ids.append('spk-%s' % str(spk_lbs[i]))
    spk_ids = np.asarray(spk_ids, dtype=unicode)

    # Save X, spk_ids, n_frames, spk_path to outfile
    os.remove(outfile) if os.path.isfile(outfile) else None
    print('Saving data to %s' % outfile)
    with h5.File(outfile, 'w') as f:
        f['X'] = X
        f['spk_ids'] = spk_ids
        f['n_frames'] = n_frames
        f['spk_path'] = spk_path

    return spk_lbs, scr_mat, dist_mat, sim_mat


# Regroup the entries in the pairwase score matrix according to speaker labels
def regroup_matrix(mat, lbs):
    # Group the rows having the same label
    temp = []
    for l in range(np.max(lbs)+1):
        temp.append(mat[lbs == l, :])
    reg_mat = np.concatenate(temp, axis=0)

    # Group the columns having the same label
    temp = []
    for l in range(np.max(lbs)+1):
        temp.append(reg_mat[:, lbs == l])
    reg_mat = np.concatenate(temp, axis=1)
    return reg_mat


def load_data(datafiles):
    X, spk_ids, n_frames, spk_path = [], [], [], []
    for file in datafiles:
        with h5.File(file) as f:
            print('Loading i-vector file: %s' % file)
            X.append(f['X'][:])
            spk_ids.append(f['spk_ids'][:])
            n_frames.append(f['n_frames'][:])
            spk_path.append(f['spk_path'][:])
    X = np.concatenate(X, axis=0)
    spk_ids = np.concatenate(spk_ids, axis=0)
    n_frames = np.concatenate(n_frames, axis=0)
    spk_path = np.concatenate(spk_path, axis=0)
    return X, spk_ids, n_frames, spk_path


def save_data(datafile, X, spk_ids, n_frames, spk_path):
    unicode = h5.special_dtype(vlen=str)
    with h5.File(datafile, 'w') as f:
        f['X'] = X
        f['spk_ids'] = spk_ids,
        f['n_frames'] = n_frames
        f['spk_path'] = spk_path
        f['spk_ids'] = np.array(spk_ids, dtype=unicode)


def lbs_as_matrix(lbs, n_clusters):
    _, ucount = np.unique(lbs, return_counts=True)
    n_vecs = len(lbs)
    lbs_mat = np.zeros((n_vecs, n_vecs))
    i = 0
    for k in range(n_clusters):
        idx = range(i, i + ucount[k])
        lbs_mat[idx[0]: idx[-1], idx[0]:idx[-1]] = 1
        i = i + ucount[k]
    return lbs_mat


def make_spkids(labels):
    spk_ids = []
    for label in labels:
        spk_ids.append('spk-' + str(label))
    return spk_ids


####################################
# The beginning of the main part
####################################

def main():
    if len(sys.argv) != 2:
        print('Usage: %s <prep_mode>' % sys.argv[0])
        print('       Valid prep_mode: lennorm|whiten+lennorm|wccn+lennorm+wccn|wccn+lennorm+lda+wccn')
        exit()

    # Define the preprocessing mode and data file
    prep_mode = sys.argv[1]
    domain = 'sre16-dev'
    # domain = 'sre05-12'
    sc_outfile = 'data/h5/sre16_dev_sclabeled_t300_w_512c.h5'
    ac_outfile = 'data/h5/sre16_dev_aclabeled_t300_w_512c.h5'
    modelfile = 'data/h5/model/plda-%s.h5' % prep_mode
    scrmatfile = 'data/h5/plda-scoremat.h5'
    partial = False              # Compute the first 200x200 entries in the score matrix only
    n_hyspks = 200               # No. of hypothesized speakers
    sigma = 100

    # Set input files
    if domain == 'sre16-dev':
        infiles = ['data/h5/sre16_dev_unlabeled_t300_w_512c.h5']
        outfile_sc = 'data/h5/sre16_dev_sclabeled_t300_w_512c.h5'
        outfile_ac = 'data/h5/sre16_dev_aclabeled_t300_w_512c.h5'
    else:
        infiles = ['./data/h5/male_plda_checked_t300_w_512c-1.h5']

    # Perform speaker clustering
    sc_lbs, scr_mat, dist_mat, sim_mat = \
        speaker_clustering(infiles, sc_outfile, modelfile, scr_mat=None, partial=partial,
                           prep_mode=prep_mode, clus_method='sc', n_clusters=n_hyspks, sigma=sigma)
    ac_lbs, scr_mat, dist_mat, sim_mat = \
        speaker_clustering(infiles, ac_outfile, modelfile, scr_mat=scr_mat, partial=partial,
                           prep_mode=prep_mode, clus_method='ac', n_clusters=n_hyspks, sigma=sigma)

    # Save the score matrix file for future used
    os.remove(scrmatfile) if os.path.isfile(scrmatfile) else None
    save_scoremat(scrmatfile, scr_mat)

    # Make spk_ids by appending 'spk-' to the labels
    sc_spk_ids = make_spkids(sc_lbs)
    ac_spk_ids = make_spkids(ac_lbs)

    # Plot clustering results
    # Compute the Silhouette values of the clusters based on the labels found by clustering
    distmat = np.max(np.abs(sim_mat)) - sim_mat
    sc_sil_val = silhouette_samples(distmat, sc_lbs, metric='precomputed')
    sc_sil_avg = silhouette_score(distmat, sc_lbs, metric='precomputed')
    ah_sil_val = silhouette_samples(distmat, ac_lbs, metric='precomputed')
    ah_sil_avg = silhouette_score(distmat, ac_lbs, metric='precomputed')

    # Regrouping rows and cols of matrix according to the labels
    sc_rgp_simmat = regroup_matrix(sim_mat, sc_lbs)
    ah_rgp_simmat = regroup_matrix(sim_mat, ac_lbs)

    # Create a figure
    plt.figure(figsize=(15, 18))

    # Plot the original PLDA score matrix as image
    plt.subplot(3, 3, 1)
    np.fill_diagonal(scr_mat, np.min(scr_mat))
    imshow(scr_mat)
    plt.title('Original Scores Matrix')
    plt.colorbar()

    # Plot the original PLDA distance matrix as image
    plt.subplot(3, 3, 2)
    np.fill_diagonal(sim_mat, 0.0)
    imshow(dist_mat)
    plt.title('Original Distance Matrix')
    plt.colorbar()

    # Plot the original similarity matrix as image
    plt.subplot(3, 3, 3)
    np.fill_diagonal(sim_mat, 0.0)
    imshow(sim_mat)
    plt.title('Original Similarity Matrix')
    plt.colorbar()

    # Plot the sc-regrouped similarity matrix as image
    plt.subplot(3, 3, 4)
    np.fill_diagonal(sc_rgp_simmat, 0.0)
    imshow(sc_rgp_simmat)
    plt.title('SC-Regrouped Similarity Matrix')
    plt.colorbar()

    # Plot class labels as image
    plt.subplot(3, 3, 5)
    imshow(lbs_as_matrix(sc_lbs, n_hyspks))
    plt.title('SC Cluster Labels')

    # Produce a Silhouette plot
    plt.subplot(3, 3, 6)
    plt.barh(np.arange(len(sc_lbs)), sc_sil_val)
    plt.xlabel('Silhouette Value')
    plt.ylabel('Sample Index')
    plt.title('SC Average Silhouette Score = %.2f' % sc_sil_avg)
    plt.axis([-1, 1, 0, len(sc_lbs)])

    # Plot the ahc-regrouped similarity matrix as image
    plt.subplot(3, 3, 7)
    np.fill_diagonal(ah_rgp_simmat, 0.0)
    imshow(ah_rgp_simmat)
    plt.title('AHC-Regrouped Similarity Matrix')
    plt.colorbar()

    # Plot class labels as image
    plt.subplot(3, 3, 8)
    imshow(lbs_as_matrix(ac_lbs, n_hyspks))
    plt.title('AHC Cluster Labels')

    # Produce a Silhouette plot
    plt.subplot(3, 3, 9)
    plt.barh(np.arange(len(ac_lbs)), ah_sil_val)
    plt.xlabel('Silhouette Value')
    plt.ylabel('Sample Index')
    plt.title('AHC Average Silhouette Score = %.2f' % ah_sil_avg)
    plt.axis([-1, 1, 0, len(ac_lbs)])

    plt.subplots_adjust(hspace=0.5)

    # Save the plot to .png file
    print('Saving fig/%s_nclus%s.png' % (domain, n_hyspks))
    plt.savefig('fig/%s_nclus%s' % (domain, n_hyspks))

    plt.show()


if __name__ == '__main__':
    main()



