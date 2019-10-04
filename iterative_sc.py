import numpy as np
# import prettytable as pt
import pprint
import networkx as nx
import scipy.io as sio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def fit_and_plot(algorithm,title):
    col = ['bo','ro','co', 'mo','ko']
    algorithm.fit(scoremat)
    n_clusters = algorithm.n_clusters
    lab = algorithm.labels_
    reds = lab == 0
    blues = lab == 1
    for jj in range(n_clusters):
        plt.plot(ivec_prj[lab == jj, 0], ivec_prj[lab == jj, 1], col[jj])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.axes().set_aspect('equal')

from sklearn import cluster, datasets
np.random.seed(0)
n_samples = 66
# X, y = datasets.make_circles(n_samples=n_samples, factor=.4, noise=.1)
scoremat_file = sio.loadmat('data/nontst_scoremat.mat')
scoremat = np.array(scoremat_file['scoremat'], dtype='float32')

ivec_file = sio.loadmat('data/extract_i_vec.mat')
ivec = np.array(ivec_file['X'], dtype='float32')
ivec_prj = TSNE(random_state=20150101).fit_transform(ivec)

spectralnn4 = cluster.SpectralClustering(n_clusters=3,
                                eigen_solver='arpack',
                                affinity='nearest_neighbors',
                                n_neighbors=22)
fit_and_plot(spectralnn4,"Spectral clustering on two circles")



W = spectralnn4.affinity_matrix_
G=nx.from_scipy_sparse_matrix(W)
nx.draw(G,ivec_prj,node_color='g',node_size=180)
plt.axis('equal')
plt.show()