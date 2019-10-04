# Train an AAE to produce gender-indepdent encoded i-vectors.
# Use the AdversarialTx class
# Implement the adversarial i-vector transformer in 2017/18 GRF proposal (Fig. 3)

# To run this script using Python3.6 (enmcomp3,4,11),
# assuming that Anaconda3 environment "tf-py3.6"
# has been created already
#   bash
#   export PATH=/usr/local/anaconda3/bin:/usr/local/cuda-8.0/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
#   source activate tf-py3.6
#   python3 cmn_adv_tran.py
#   source deactivate tf-py3.6

# M.W. Mak, Oct. 2017

from __future__ import print_function

import tensorflow as tf
from myPlot import scatter2D
import matplotlib.pyplot as plt
from ivec_utils import load_ivectors, get_filenames2, extract_data
from AdversarialTx import AdversarialTx
import numpy as np
from sklearn.manifold import TSNE
from keras.utils import np_utils
import scipy.io as sio


# Main function
def main():
    # Define some constants
    latent_dim = 300
    datadir = 'data/'
    logdir = 'log/'
    n_fac = 300
    n_gauss = 512
    n_epochs = 150
    eint = 25              # Epoch interval at
    n_tst_spks = 20         # No. of test speakers, must be an even number
    min_n_trn_vecs = 10
    min_n_tst_vecs = 30
    min_n_vecs = min_n_trn_vecs + min_n_tst_vecs        # Min. no. of i-vectors per speaker

    # Load training and test data
    matfiles, gender = get_filenames2(datadir, n_gauss, n_fac)
    x_dat, sex_dat, spk_dat = load_ivectors(matfiles, gender, shuffle=False)

    # Select speakers with at least min_n_ivecs i-vectors (sex_trn==0 for male and ==1 for female)
    m_x, m_spk_lbs = select_speakers(x_dat, spk_dat, sex_dat, gender='male', min_n_vecs=min_n_vecs, n_spks=-1)
    n_male_spks = np.max(m_spk_lbs) + 1
    f_x, f_spk_lbs = select_speakers(x_dat, spk_dat, sex_dat, gender='female', min_n_vecs=min_n_vecs, n_spks=-1)
    n_female_spks = np.max(f_spk_lbs) + 1
    f_spk_lbs = [i + n_male_spks for i in f_spk_lbs]        # Speaker labels start from 0 and consecutive

    # Prepare training and test data (input and target)
    x = np.vstack([m_x, f_x])
    spk_lbs = np.hstack([m_spk_lbs, f_spk_lbs])
    sex_lbs = np.asarray([0]*len(m_spk_lbs) + [1]*len(f_spk_lbs))
    n_spks = np.max(spk_lbs) + 1


    # trial = sio.loadmat('/home7b/lxli/matlab_mPLDA/mat/fw60/sre16_eval_tstutt_t500_w_1024c.mat')
    # sre16_enroll = sio.loadmat('/home7b/lxli/matlab_mPLDA/mat/fw60/SRE16_enroll+major.mat')
    # all_enc = np.empty((0, 300), dtype='float32')
    # trial_enc = np.empty((0, 300), dtype='float32')
    # enroll_enc = np.empty((0, 300), dtype='float32')


    # Prepare training data
    x_trn = np.empty((0, x.shape[1]), dtype='float32')
    spk_lbs = spk_lbs.tolist()
    spk_lbs_trn = list()
    sex_lbs_trn = list()
    for spk in range(n_spks):
        idx = [t for t, e in enumerate(spk_lbs) if e == spk]
        idx_trn = idx[0:min_n_trn_vecs]
        x_trn = np.vstack([x_trn, x[idx_trn, :]])
        spk_lbs_trn.extend([spk] * len(idx_trn))
        sex_lbs_trn.extend(sex_lbs[idx_trn].tolist())

    # Shuffle the training data for training AAE
    ridx = np.random.permutation(x_trn.shape[0])
    x_trn = x_trn[ridx, :]
    spk_lbs_trn = np.asarray(spk_lbs_trn)[ridx]
    spk_1h_trn = np_utils.to_categorical(spk_lbs_trn)
    sex_lbs_trn = np.asarray(sex_lbs_trn)[ridx]
    sex_1h_trn = np_utils.to_categorical(sex_lbs_trn)

    # Only display n_tst_spk test speakers (half malle and half female)
    x_tst = np.empty((0, x.shape[1]), dtype='float32')
    spk_lbs_tst = list()
    sex_lbs_tst = list()
    for m_spk in range(int(n_tst_spks/2)):                           # Select n_tst_spks/2 male speakers
        idx = [t for t, e in enumerate(spk_lbs) if e == m_spk]
        idx_tst = idx[min_n_trn_vecs:]                          # Ensure tst vectors are not the same as trn vec
        x_tst = np.vstack([x_tst, x[idx_tst, :]])
        spk_lbs_tst.extend([m_spk] * len(idx_tst))
        sex_lbs_tst.extend(sex_lbs[idx_tst].tolist())
    c = int(n_tst_spks/2)
    for f_spk in range(n_male_spks, n_male_spks + int(n_tst_spks/2)): # Select n_tst_spks/2 female speakers
        idx = [t for t, e in enumerate(spk_lbs) if e == f_spk]
        idx_tst = idx[min_n_trn_vecs:]
        x_tst = np.vstack([x_tst, x[idx_tst, :]])
        spk_lbs_tst.extend([c] * len(idx_tst))
        sex_lbs_tst.extend(sex_lbs[idx_tst].tolist())
        c = c + 1
    spk_lbs_tst = np.asarray(spk_lbs_tst)
    sex_lbs_tst = np.asarray(sex_lbs_tst)

    print('No. of training vectors = %d' % x_trn.shape[0])
    print('No. of male speakers = %d' % n_male_spks)
    print('No. of female speakers = %d' % n_female_spks)

    # Create an AAE object and train the AAE
    ad_tx = AdversarialTx(f_dim=x_trn.shape[1], z_dim=latent_dim, n_cls=np.max(spk_lbs_trn) + 1)

    # Iteratively train the AAE. Display results for every eint
    for it in range(int(n_epochs/eint)):
        epoch = (it + 1)*eint
        ad_tx.train(x_trn, spk_1h_trn, sex_1h_trn, n_epochs=eint, batch_size=128)

        # Encode the test i-vectors
        encoder = ad_tx.get_encoder()
        x_tst_enc = encoder.predict(x_tst)

        # all_enc = np.vstack((all_enc, x_x_tst_enc))


        # Plot the encoded vectors
        if latent_dim > 2:
            print('Creating t-SNE plot')
            x_tst_enc_prj = TSNE(random_state=20150101).fit_transform(x_tst_enc)
        else:
            x_tst_enc_prj = x_tst_enc
        fig, _, _, _ = scatter2D(x_tst_enc_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
                                 title='Adversarially Transformed I-Vectors (Epoch = %d)' % epoch)
        filename = logdir + 'aae4-%d-%d-epoch%d.png' % (latent_dim, min_n_vecs, epoch)
        fig.savefig(filename)
        plt.show(block=False)


    # all_enc = encoder.predict(x_dat)
    # sio.savemat('data/encoded_weight.mat', {'w': all_enc})
    # trial_enc = encoder.predict(trial['w'])
    # sio.savemat('data/encoded_tst_SRE16.mat',
    #             {'w': trial_enc, 'spk_logical': trial['spk_logical'], 'spk_physical': trial['spk_physical'],
    #              'num_frames': trial['num_frames']})
    # enroll_enc = encoder.predict(sre16_enroll['w'])
    # sio.savemat('data/encoded_enroll_SRE16.mat',
    #             {'w': enroll_enc, 'spk_logical': sre16_enroll['spk_logical'], 'L': sre16_enroll['L']})



    # Use t-SNE to plot the i-vectors on 2-D space
    print('Creating t-SNE plot')
    x_prj = TSNE(random_state=20150101).fit_transform(x_tst)
    fig, _, _, _ = scatter2D(x_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
                             title='Original i-vectors')
    filename = logdir + 'ivec-%d.png' % n_fac
    fig.savefig(filename)
    plt.show()


def select_speakers(x, spk_lbs, gender_lbs, gender='male', min_n_vecs=40, n_spks=-1):
    lbs = 0 if gender == 'male' else 1
    idx = [t for t, e in enumerate(gender_lbs) if e == lbs]
    sel_x, sel_spk_lbs = extract_data(x[idx, :], spk_lbs[idx], min_n_vecs=min_n_vecs, n_spks=n_spks, shuffle=False)
    return sel_x, sel_spk_lbs


if __name__ == '__main__':
    # Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    np.random.seed(1)

    main()


