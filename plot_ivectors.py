"""
Use t-SNE to plot the i-vectors from pre-SRE16 and SRE16
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import h5py as h5
import seaborn as sns
import itertools
import sys
from scipy import linalg as la
from lib.h5helper import h52dict
from lib.my_func import lennorm
from speaker_clustering import get_plda_matrix, score2sim
from lib.preprocess import preprocess


# Load I-vectors from i-vector files
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
    data = {'X': X, 'spk_ids': spk_ids, 'n_frames': n_frames, 'spk_path': spk_path}
    return data


# Here is a utility function used to display the transformed dataset.
# The color of each point refers to the actual digit (of course,
# this information was not used by the dimensionality reduction algorithm).
# For general classification problem (not MNIST digit recognition), colors
# contain the class labels
def scatter2D(x, colors, legends):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    markers = itertools.cycle(('*', '<', 's', 'o', 'D', 'X'))
    n_colors = len(np.unique(colors))

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_colors))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(n_colors):
        sc = ax.scatter(x[colors == i, 0], x[colors == i, 1], lw=0, s=10,
                        c=palette[colors[colors == i].astype(np.int)],
                        marker=markers.__next__(), label=legends[i])
    ax.grid(b=False)
    leg = ax.legend(loc='upper right', frameon=True)
    leg.get_frame().set_edgecolor('b')
    ax.axis('tight')
    plt.xlim(-105, 120)
    plt.ylim(-97, 139)

    return f, ax, sc


# Compute the pairwise distances between the means of processed i-vectors
def get_distmatrix(Xlist):
    dim = Xlist[0].shape[1]
    mu = np.zeros((len(Xlist), dim))
    for i in range(len(Xlist)):
        mu[i] = Xlist[i].mean(0)
    distmat = np.zeros((len(Xlist), len(Xlist)))
    for i in range(len(Xlist)):
        for j in range(len(Xlist)):
            distmat[i][j] = la.norm(mu[i] - mu[j])
    return distmat


# Compute the sum of the eigenvalues of the covariance matrices
def get_eigenvaluesum(Xlist):
    ev_sum = np.zeros((len(Xlist), ))
    for i in range(len(Xlist)):
        covmat = np.cov(Xlist[i].T)
        covmat = (covmat + covmat.T)/2          # Ensure simmetric, so no complex eigenvalues
        ev_sum[i] = np.sum(np.real(np.linalg.eigvals(covmat)))
    return ev_sum


def remove_bad_ivectors(X):
    mask = []
    X_norm = la.norm(X, axis=1)
    mask.append(X_norm < 60.0)
    return X[mask]


# Plot the norm of all vectors
def plot_ivec_norm(Xlist, dataname):
    colors = itertools.cycle(('b', 'r', 'k', 'g', 'c', 'm'))
    t_start = 0
    plt.figure()
    ax = plt.subplot()
    for i in range(len(Xlist)):
        normX = np.linalg.norm(Xlist[i], axis=1)
        t_end = t_start + normX.shape[0]
        plt.plot(np.arange(t_start, t_end), normX, colors.__next__(), label=dataname[i])
        t_start = t_end
    leg = ax.legend(loc='upper right', frameon=True)
    leg.get_frame().set_edgecolor('b')
    plt.xlabel('I-Vector Index')
    plt.ylabel('I-vector Norm')


def main():
    if len(sys.argv) != 2:
        print('Usage: %s <prep_mode>' % sys.argv[0])
        print('       Valid prep_mode: lennorm|whiten+lennorm|wccn+lennorm+wccn|wccn+lennorm+lda+wccn')
        exit()

    # Options and constants
    prep_mode = sys.argv[1]
    modelfile = 'data/h5/model/plda-%s.h5' % prep_mode
    dev_unlabel = ['data/h5/sre16_dev_unlabeled_t300_w_512c.h5']
    dev_test = ['./data/h5/sre16_dev_tstutt_t300_w_512c.h5']
    evl_test = ['./data/h5/sre16_eval_tstutt_t300_w_512c.h5']
    sc_label = ['data/h5/sre16_dev_sclabeled_t300_w_512c.h5']
    ac_label = ['data/h5/sre16_dev_aclabeled_t300_w_512c.h5']
    dev_enroll = ['data/h5/sre16_dev_enrollments_t300_w_512c.h5']
    evl_enroll = ['data/h5/sre16_eval_enrollments_t300_w_512c.h5']
    pre16 = ['data/h5/female_plda_checked_t300_w_512c-1.h5']
    N = 5000            # Max. no. of sample per datafile

    # Load i-vector files from different sources
    sources = [pre16, dev_enroll, dev_unlabel, dev_test, evl_enroll, evl_test]
    data = []
    for i in range(len(sources)):
        data.append(load_data(sources[i]))

    # Load PLDA model
    print('Loading model: %s' % modelfile)
    plda = h52dict(modelfile)

    # Pre-process i-vectors and subtract global mean as defined in PLDA model
    Xlist = []
    for i in range(len(sources)):
        data[i]['X'] = preprocess(data[i]['X'], model=plda, mode=prep_mode)
        Xlist.append(data[i]['X'])

    # Print distances between means
    np.set_printoptions(precision=3)
    print(np.matrix(get_distmatrix(Xlist)*100))

    # Print the sum of the eigenvalues of the covariance matrices from different sources
    print(np.array(get_eigenvaluesum(Xlist)*1000))

    # Extract N i-vectors from each source
    Xlist = []
    for i in range(len(sources)):
        data[i]['X'] = data[i]['X'][0:N, :]
        data[i]['spk_id'] = data[i]['X'][0:N]
        data[i]['n_frames'] = data[i]['X'][0:N]
        data[i]['spk_path'] = data[i]['X'][0:N]
        Xlist.append(data[i]['X'])

    # Assign labels
    labels = []
    for i in range(len(Xlist)):
        M = np.min([N, Xlist[i].shape[0]])
        labels.append(np.ones((M, )) * i)
    labels = np.concatenate(labels, axis=0)
    labels = np.asarray(labels, dtype=int)
    dataname = ['SRE05-12', 'SRE16-dev-enroll', 'SRE16-dev-unlabel',
                'SRE16-dev-test', 'SRE16-eval-enroll', 'SRE16-eval-test']

    # Plot the norm of all vectors
    plot_ivec_norm(Xlist, dataname)
    plt.show(block=False)

    # Compute PLDA distance matrix for t-SNE
    X = np.concatenate(Xlist, axis=0)
    scrmat = get_plda_matrix(X, plda, prep_mode, partial=False)

    # Convert score matrix to similarity and distance matrix
    _, distmat = score2sim(scrmat, sigma=100)

    # Project i-vectors using t-SNE
    print('Performing t-SNE transformation...')
    X_prj = TSNE(random_state=20150101, metric='precomputed').fit_transform(distmat)

    # Plot the norm of all t-SNE projected vectors
    t_start = 0
    for i in range(len(Xlist)):
        t_end = t_start + Xlist[i].shape[0]
        Xlist[i] = X_prj[t_start:t_end, :]
        t_start = t_end
    plot_ivec_norm(Xlist, dataname)
    plt.show(block=False)

    # Produce t-SNE scatter plot
    scatter2D(X_prj, labels, dataname)
    plt.show(block=True)


if __name__ == '__main__':
    main()
