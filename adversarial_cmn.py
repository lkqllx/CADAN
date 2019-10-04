from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from myPlot import scatter2D
from ivec_utilis_sex import load_ivectors, get_filenames1, extract_data
from sklearn.manifold import TSNE
from keras.utils import np_utils
import pickle

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def select_speakers(x, spk_lbs, gender_lbs, gender='male', min_n_vecs=50, n_spks=-1):
    lbs = 0 if gender == 'male' else 1
    idx = [t for t, e in enumerate(gender_lbs) if e == lbs]
    sel_x, sel_spk_lbs, out_x, out_spk_lbs = extract_data(x[idx, :], spk_lbs[idx], min_n_vecs=min_n_vecs, n_spks=n_spks, shuffle=False)
    return sel_x, sel_spk_lbs, out_x, out_spk_lbs

# Define some constants
latent_dim = 500
datadir = 'data/'
logdir = 'log/'
n_fac = 500
n_gauss = 1024
n_epochs = 3000
eint = 100              # Epoch interval at
n_tst_spks = 20         # No. of test speakers, must be an even number
min_n_trn_vecs = 20
min_n_tst_vecs = 20
min_n_vecs = min_n_trn_vecs + min_n_tst_vecs        # Min. no. of i-vectors per speaker

# Load training and test data
matfiles, gender = get_filenames1(datadir, n_gauss, n_fac)
x_dat, sex_dat, spk_dat = load_ivectors(matfiles, gender, shuffle=False)

# # Select speakers with at least min_n_ivecs i-vectors (sex_trn==0 for male and ==1 for female)
m_x, m_spk_lbs, out_m_x, out_m_spk_lbs = select_speakers(x_dat, spk_dat, sex_dat, gender='male', min_n_vecs=min_n_vecs, n_spks=-1)
out_n_male_spks = np.max(out_m_spk_lbs) + 1
n_male_spks = np.max(m_spk_lbs) + 1
#
f_x, f_spk_lbs, out_f_x, out_f_spk_lbs  = select_speakers(x_dat, spk_dat, sex_dat, gender='female', min_n_vecs=min_n_vecs, n_spks=-1)
n_female_spks = np.max(f_spk_lbs) + 1
out_n_female_spks = np.max(out_f_spk_lbs) + 1
f_spk_lbs = [i + n_male_spks for i in f_spk_lbs]        # Speaker labels start from 0 and consecutive
out_f_spk_lbs = [i + out_n_male_spks for i in out_f_spk_lbs]
# Load training and test data


# Prepare training and test data (input and target)
x = np.vstack([m_x, f_x])
spk_lbs = np.hstack([m_spk_lbs, f_spk_lbs])
sex_lbs = np.asarray([0]*len(m_spk_lbs) + [1]*len(f_spk_lbs))
n_spks = np.max(spk_lbs) + 1


all_male = np.vstack([m_x])
all_male_spk_lbs = np.hstack([m_spk_lbs])
all_male_sex_lbs = np.asarray([0]*len(m_spk_lbs) )
all_male_n_spks = np.max(all_male_spk_lbs) + 1

all_female = np.vstack([f_x])
all_female_spk_lbs = np.hstack([f_spk_lbs])
all_female_sex_lbs = np.asarray([0]*len(f_spk_lbs))
all_female_n_spks = np.max(all_female_spk_lbs) + 1
# all_male = np.vstack([m_x, out_m_x])
# all_male_spk_lbs = np.hstack([m_spk_lbs, out_m_spk_lbs])
# all_male_sex_lbs = np.asarray([0]*len(m_spk_lbs) + [1]*len(out_m_spk_lbs))
# all_male_n_spks = np.max(all_male_spk_lbs) + 1
#
# all_female = np.vstack([f_x, out_f_x])
# all_female_spk_lbs = np.hstack([f_spk_lbs, out_f_spk_lbs])
# all_female_sex_lbs = np.asarray([0]*len(f_spk_lbs) + [1]*len(out_f_spk_lbs))
# all_female_n_spks = np.max(all_female_spk_lbs) + 1


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
x_tst_male = np.empty((0, x.shape[1]), dtype='float32')
spk_lbs_tst_male = list()
sex_lbs_tst_male = list()
x_tst_female = np.empty((0, x.shape[1]), dtype='float32')
spk_lbs_tst_female = list()
sex_lbs_tst_female = list()
for m_spk in range(int(n_tst_spks/2)):                           # Select n_tst_spks/2 male speakers
    idx = [t for t, e in enumerate(spk_lbs) if e == m_spk]
    idx_tst = idx[min_n_trn_vecs:]                          # Ensure tst vectors are not the same as trn vec
    x_tst = np.vstack([x_tst, x[idx_tst, :]])
    spk_lbs_tst.extend([m_spk] * len(idx_tst))
    sex_lbs_tst.extend(sex_lbs[idx_tst].tolist())
c = int(n_tst_spks/2)
#
# for m_spk in range(int(np.max(m_spk_lbs + 1))):                           # Select n_tst_spks/2 male speakers
#     idx = [t for t, e in enumerate(spk_lbs) if e == m_spk]
#     idx_tst = idx[min_n_trn_vecs:]                          # Ensure tst vectors are not the same as trn vec
#     x_tst_male= np.vstack([x_tst_male, x[idx_tst, :]])
#     spk_lbs_tst_male.extend([m_spk] * len(idx_tst))
#     sex_lbs_tst_male.extend(sex_lbs[idx_tst].tolist())


for f_spk in range(n_male_spks, n_male_spks + int(n_tst_spks/2)): # Select n_tst_spks/2 female speakers
    idx = [t for t, e in enumerate(spk_lbs) if e == f_spk]
    idx_tst = idx[min_n_trn_vecs:]
    x_tst = np.vstack([x_tst, x[idx_tst, :]])
    spk_lbs_tst.extend([c] * len(idx_tst))
    sex_lbs_tst.extend(sex_lbs[idx_tst].tolist())
    c = c + 1

# for f_spk in range(int(np.max(f_spk_lbs + 1))):                           # Select n_tst_spks/2 male speakers
#     idx = [t for t, e in enumerate(spk_lbs) if e == f_spk]
#     idx_tst = idx[min_n_trn_vecs:]                          # Ensure tst vectors are not the same as trn vec
#     x_tst_female= np.vstack([x_tst_female, x[idx_tst, :]])
#     spk_lbs_tst_female.extend([m_spk] * len(idx_tst))
#     sex_lbs_tst_female.extend(sex_lbs[idx_tst].tolist())

spk_lbs_tst = np.asarray(spk_lbs_tst)
sex_lbs_tst = np.asarray(sex_lbs_tst)

num_hidden_D = 500
num_hidden_G = 1800
num_hidden_C = 500
dim_i_vector = 500

D_W1 = tf.Variable(xavier_init([500, num_hidden_D]))
D_b1 = tf.Variable(tf.zeros(shape=[num_hidden_D]))
D_W2 = tf.Variable(xavier_init([num_hidden_D, num_hidden_D]))
D_b2 = tf.Variable(tf.zeros(shape=[num_hidden_D]))
D_W3 = tf.Variable(xavier_init([num_hidden_D, 2]))
D_b3 = tf.Variable(tf.zeros(shape=[2]))

G_W1 = tf.Variable(xavier_init([500, num_hidden_G]))
G_b1 = tf.Variable(tf.zeros(shape=[num_hidden_G]))
G_W2 = tf.Variable(xavier_init([num_hidden_G, num_hidden_G]))
G_b2 = tf.Variable(tf.zeros(shape=[num_hidden_G]))
G_W3 = tf.Variable(xavier_init([num_hidden_G, num_hidden_G]))
G_b3 = tf.Variable(tf.zeros(shape=[num_hidden_G]))
G_W4 = tf.Variable(xavier_init([num_hidden_G, 500]))
G_b4 = tf.Variable(tf.zeros(shape=[500]))

C_W1 = tf.Variable(xavier_init([500, num_hidden_C]))
C_b1 = tf.Variable(tf.zeros(shape=[num_hidden_C]))
C_W2 = tf.Variable(xavier_init([num_hidden_C, num_hidden_C]))
C_b2 = tf.Variable(tf.zeros(shape=[num_hidden_C]))
C_W3 = tf.Variable(xavier_init([num_hidden_C, spk_1h_trn.shape[1]]))
C_b3 = tf.Variable(tf.zeros(shape=[spk_1h_trn.shape[1]]))

# def batch_norm_2(x, phase):
#
#         h1 = tf.contrib.layers.batch_norm(x,
#                                           center=True, scale=True, epsilon = 0.001, decay=0.5,
#                                           is_training=phase)
#         return h1
#
# def generator(x, train_flag = 1):
#
#     x_norm = batch_norm_2(x, train_flag)
#     G_h1 = tf.nn.relu(tf.matmul(x_norm, G_W1) + G_b1)
#     G_h1_norm = batch_norm_2(G_h1, train_flag)
#     G_h2= tf.matmul(G_h1_norm, G_W2) + G_b2
#     G_h2_norm = batch_norm_2(G_h2, train_flag)
#     G_log_prob= tf.matmul(G_h2_norm, G_W3) + G_b3
#     G_prob = tf.nn.sigmoid(G_log_prob)
#
#     return G_prob
#
#
# def discriminator(z, train_flag = 1):
#     z_norm = batch_norm_2(z, train_flag)
#     D_h1 = tf.nn.relu(tf.matmul(z_norm, D_W1) + D_b1)
#     D_h1_norm = batch_norm_2(D_h1, train_flag)
#     D_h2 = tf.nn.relu(tf.matmul(D_h1_norm, D_W2) + D_b2)
#     D_h2_norm = batch_norm_2(D_h2, train_flag)
#     D_logit = tf.matmul(D_h2_norm, D_W3) + D_b3
#     D_prob = tf.nn.softmax(D_logit)
#     return D_prob, D_logit
#
# def classifier(z, train_flag = 1):
#     z_norm = batch_norm_2(z, train_flag)
#     C_h1 = tf.nn.relu(tf.matmul(z_norm, C_W1) + C_b1)
#     C_h1_norm = batch_norm_2(C_h1, train_flag)
#     C_h2 = tf.nn.relu(tf.matmul(C_h1_norm, C_W2) + C_b2)
#     C_h2_norm = batch_norm_2(C_h2, train_flag)
#     C_logit = tf.matmul(C_h2_norm, C_W3) + C_b3
#     C_prob = tf.nn.softmax(C_logit)
#     return C_prob, C_logit

def batch_norm(Wx_plus_b, out_size, train_flag):
    # if train_flag == True:
        # Batch Normalize
    fc_mean, fc_var = tf.nn.moments(
        Wx_plus_b,
        axes=[0]  # the dimension you wanna normalize, here [0] for batch
        # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
    )
    scale = tf.Variable(tf.ones(out_size))
    shift = tf.Variable(tf.zeros(out_size))
    epsilon = 0.001
    # apply moving average for mean and var when train on batch
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)

    mean, var = mean_var_with_update()
    # else:
    #
    #     def test_mean_var_with_update():
    #         return tf.identity(fc_mean), tf.identity(fc_var)
    #     scale = tf.Variable(tf.ones(out_size))
    #     shift = tf.Variable(tf.zeros(out_size))
    #     epsilon = 0.001
    #     mean, var = test_mean_var_with_update()

    return mean, var, shift, scale, epsilon


def generator(x, test_flag = False):
    mean, var, shift, scale, epsilon = batch_norm(x, dim_i_vector, test_flag)
    x_norm = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

    G_h1 = tf.matmul(x_norm, G_W1) + G_b1
    mean, var, shift, scale, epsilon = batch_norm(G_h1, num_hidden_G, test_flag)
    G_h1_norm = tf.nn.batch_normalization(G_h1, mean, var, shift, scale, epsilon)
    G_h1_norm = tf.nn.relu(G_h1_norm)

    G_h2= tf.matmul(G_h1_norm, G_W2) + G_b2
    mean, var, shift, scale, epsilon = batch_norm(G_h2, num_hidden_G, test_flag)
    G_h2_norm = tf.nn.batch_normalization(G_h2, mean, var, shift, scale, epsilon)
    G_h2_norm = tf.nn.relu(G_h2_norm)

    G_h3= tf.matmul(G_h2_norm, G_W3) + G_b3
    mean, var, shift, scale, epsilon = batch_norm(G_h3, num_hidden_G, test_flag)
    G_h3_norm = tf.nn.batch_normalization(G_h3, mean, var, shift, scale, epsilon)
    G_h3_norm = tf.nn.relu(G_h3_norm)

    G_log_prob= tf.matmul(G_h3_norm, G_W4) + G_b4
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(z, test_flag = False):
    mean, var, shift, scale, epsilon = batch_norm(z, dim_i_vector, test_flag)
    z_norm = tf.nn.batch_normalization(z, mean, var, shift, scale, epsilon)

    D_h1 = tf.matmul(z_norm, D_W1) + D_b1
    mean, var, shift, scale, epsilon = batch_norm(D_h1, num_hidden_D, test_flag)
    D_h1_norm = tf.nn.batch_normalization(D_h1, mean, var, shift, scale, epsilon)
    D_h1_norm = tf.nn.relu(D_h1_norm)

    D_h2 = tf.matmul(D_h1_norm, D_W2) + D_b2
    mean, var, shift, scale, epsilon = batch_norm(D_h2, num_hidden_D, test_flag)
    D_h2_norm = tf.nn.batch_normalization(D_h2, mean, var, shift, scale, epsilon)
    D_h2_norm = tf.nn.relu(D_h2_norm)

    D_logit = tf.matmul(D_h2_norm, D_W3) + D_b3
    D_prob = tf.nn.softmax(D_logit)
    return D_prob, D_logit

def classifier(z, test_flag = False):
    mean, var, shift, scale, epsilon = batch_norm(z, dim_i_vector, test_flag)
    z_norm = tf.nn.batch_normalization(z, mean, var, shift, scale, epsilon)

    C_h1 = tf.matmul(z_norm, C_W1) + C_b1
    mean, var, shift, scale, epsilon = batch_norm(C_h1, num_hidden_C, test_flag)
    C_h1_norm = tf.nn.batch_normalization(C_h1, mean, var, shift, scale, epsilon)
    C_h1_norm = tf.nn.relu(C_h1_norm)

    C_h2 = tf.matmul(C_h1_norm, C_W2) + C_b2
    mean, var, shift, scale, epsilon = batch_norm(C_h2, num_hidden_C, test_flag)
    C_h2_norm = tf.nn.batch_normalization(C_h2, mean, var, shift, scale, epsilon)
    C_h2_norm = tf.nn.relu(C_h2_norm)

    C_logit = tf.matmul(C_h2_norm, C_W3) + C_b3
    C_prob = tf.nn.softmax(C_logit)
    return C_prob, C_logit

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def main():


    print('No. of training vectors = %d' % x_trn.shape[0])
    print('No. of male speakers = %d' % n_male_spks)
    print('No. of female speakers = %d' % n_female_spks)

    #Build a GAN with classfier
    X = tf.placeholder(tf.float32, shape=[None, 500])
    sex_X = tf.placeholder(tf.float32, shape=[None, 2])
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]

    spk_label = tf.placeholder(tf.float32, shape=[None, spk_1h_trn.shape[1]])
    theta_C = [C_W1, C_W2, C_W3, C_b1, C_b2, C_b3]

    train_flag = tf.placeholder(tf.bool)


    #X should be i-vector, G_encoded is the target latent vector.
    G_encoded = generator(X, train_flag)
    # G_encoded = np.asarray(G_encoded_array, np.float32)
    D, D_logit = discriminator(G_encoded, train_flag)
    C, C_logit = classifier(G_encoded, train_flag)


    # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    # G_loss = -tf.reduce_mean(tf.log(D_fake))

    # Alternative losses:
    # -------------------
    # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    #Label should be a placeholder which contains the sex_lb of mini-batch
    D_loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit, labels=sex_X))

    C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_logit, labels=(1 - spk_label)))

    G_loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit, labels=1 - sex_X))
    G_loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_logit, labels=spk_label))
    G_loss = G_loss_C + G_loss_D



    D_solver = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(0.00001).minimize(G_loss, var_list=theta_G)
    C_solver = tf.train.AdamOptimizer(0.00001).minimize(C_loss, var_list=[theta_C])
    G_C_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss_C, var_list=theta_G)
    G_D_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss_D, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')


    batch_size = 128
    n_batches = int(x_trn.shape[0] / batch_size)
    loss_list2 = []
    for it in range(351):
        pidx = np.random.permutation(x_trn.shape[0])
        for batch in range(n_batches):
            idx = pidx[batch * batch_size: (batch + 1) * batch_size]

            x = x_trn[idx]
            y_sex_1h = sex_1h_trn[idx]
            y_spk_1h = spk_1h_trn[idx]

            # for i in range(3):
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x, sex_X: y_sex_1h, train_flag: 1})
            # if D_loss_curr > 0.5:
            #     for i in range (1):
            #         _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x, sex_X: y_sex_1h, train_flag: 1})
            # for i in range(2):
            _, G_loss_curr, G_loss_C_curr, G_loss_D_curr = sess.run([G_solver, G_loss, G_loss_C, G_loss_D], feed_dict={X: x, sex_X: y_sex_1h, spk_label: y_spk_1h, train_flag: 1})
            # if D_loss_curr < 0.4:
            #     for i in range (int(G_loss_D_curr)):
            #         _, G_loss_curr, G_loss_C_curr, G_loss_D_curr = sess.run([G_D_solver, G_loss, G_loss_C, G_loss_D],
            #                                                                 feed_dict={X: x, sex_X: y_sex_1h,
            #                                                                            spk_label: y_spk_1h,
            #                                                                            train_flag: 1})
            #     for i in range (int(G_loss_C_curr)):
            #         _, G_loss_curr, G_loss_C_curr, G_loss_D_curr = sess.run([G_C_solver, G_loss, G_loss_C, G_loss_D],
            #                                                                 feed_dict={X: x, sex_X: y_sex_1h,
            #                                                                            spk_label: y_spk_1h,
            #                                                                            train_flag: 1})

            if it % 5 == 0:
                _, C_loss_curr = sess.run([C_solver, C_loss], feed_dict={X: x, spk_label: y_spk_1h, train_flag: 1})
            # for i in range(2):



            if it % 10 == 0 and batch == 1:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print('G_loss_C: {:.4}'.format(G_loss_C_curr))
                print('G_loss_D: {:.4}'.format(G_loss_D_curr))
                print('C_loss: {:.4}'.format(C_loss_curr))
                # print('D_solver: {:.4}'.format(D_logit_fake_curr))
                print()
            # fig, _, _, _ lock=False)
            if it % 500  == 0 and batch == 1 :
                # mean, var, shift, scale, epsilon = batch_norm(x_tst,  tf.float32(300))
                # x_tst_norm = tf.nn.batch_normalization(x_tst, mean, var, shift, scale, epsilon)
                x_tst_enc = sess.run(G_encoded, feed_dict={X: x_tst, train_flag: 1})
                print('Creating t-SNE plot')
                x_tst_enc_prj = TSNE(random_state=20150101).fit_transform(x_tst_enc)
                fig, _, _, _ = scatter2D(x_tst_enc_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
                                             title='Adversarially Transformed I-Vectors (Epoch = %d)' % it)
                filename = logdir + 'aae4-%d-%d-epoch%d.png' % (300, min_n_vecs, it)
                fig.savefig(filename)
                plt.show(block=False)
        loss_list2.append(G_loss_C_curr)
    print('Creating t-SNE plot')
    x_prj = TSNE(random_state=20150101).fit_transform(x_tst)
    fig, _, _, _ = scatter2D(x_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
                             title='Original i-vectors')
    filename = logdir + 'ivec-%d.png' % n_fac
    fig.savefig(filename)

    fig_loss, ax = plt.subplots()
    ax.plot(loss_list2)
    with open('loss_list2.pkl', 'wb') as f:
        pickle.dump(loss_list2, f)
    plt.show()

    # Create an AAE object and train the AAE
    # ad_tx = AdversarialTx(f_dim=x_trn.shape[1], z_dim=latent_dim, n_cls=np.max(spk_lbs_trn) + 1)
    #
    # # Iteratively train the AAE. Display results for every eint
    # for it in range(int(n_epochs/eint)):
    #     epoch = (it + 1)*eint
    #     ad_tx.train(x_trn, spk_1h_trn, sex_1h_trn, n_epochs=eint, batch_size=128)
    #
    #     # Encode the test i-vectors
    #     encoder = ad_tx.get_encoder()
    #     x_tst_enc = encoder.predict(x_tst)............................
    #
    #     # all_enc = np.vstack((all_enc, x_tst_enc))
    #
    #
    #     # Plot the encoded vectors
    #     if latent_dim > 2:
    #         print('Creating t-SNE plot')
    #         x_tst_enc_prj = TSNE(random_state=20150101).fit_transform(x_tst_enc)
    #     else:
    #         x_tst_enc_prj = x_tst_enc
    #     fig, _, _, _ = scatter2D(x_tst_enc_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
    #                              title='Adversarially Transformed I-Vectors (Epoch = %d)' % epoch)
    #     filename = logdir + 'aae4-%d-%d-epoch%d.png' % (latent_dim, min_n_vecs, epoch)
    #     fig.savefig(filename)
    #     plt.show(block=False)


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
    # print('Creating t-SNE plot')
    # x_prj = TSNE(random_state=20150101).fit_transform(x_tst)
    # fig, _, _, _ = scatter2D(x_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
    #                          title='Original i-vectors')
    # filename = logdir + 'ivec-%d.png' % n_fac
    # fig.savefig(filename)
    # plt.show()

if __name__ == '__main__':
    # Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    np.random.seed(1)

    main()

