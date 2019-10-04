from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from myPlot import scatter2D
from ivec_utils import load_ivectors, get_filenames2, extract_data
from sklearn.manifold import TSNE
from keras.utils import np_utils
import scipy.io as sio
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def select_speakers(x,  gender_lbs, spk_lbs, channel_lbs, gender='male', min_n_vecs=40, n_spks=-1):
    lbs = 0.0 if gender == 'male' else 1.0
    idx = [t for t, e in enumerate(gender_lbs) if e == lbs]
    sel_x, sel_spk_lbs, out_x, out_spk_lbs = extract_data(x[idx, :], spk_lbs[idx], min_n_vecs=min_n_vecs, n_spks=n_spks, c_lbs = channel_lbs[idx], shuffle=False)
    return sel_x, sel_spk_lbs, out_x, out_spk_lbs

# Define some constants
latent_dim = 500
datadir = 'jfa/'
logdir = 'log/'
n_fac = 300
n_gauss = 512
n_epochs = 3000
eint = 100              # Epoch interval at
n_tst_spks = 20         # No. of test speakers, must be an even number
min_n_trn_vecs = 15
min_n_tst_vecs = 10
min_n_vecs = min_n_trn_vecs + min_n_tst_vecs        # Min. no. of i-vectors per speaker

# Load training and test data
matfiles, gender, channel = get_filenames2(datadir, n_gauss, n_fac, 'mic')

x_dat, sex_dat, spk_dat, channel_lbs= load_ivectors(matfiles, gender, channel, shuffle=False)

# Select speakers with at least min_n_ivecs i-vectors (sex_trn==0 for male and ==1 for female)
m_x, m_spk_lbs, out_m_x, out_m_spk_lbs = select_speakers(x_dat, sex_dat, spk_dat, channel_lbs, gender='male', min_n_vecs=30, n_spks=-1)
out_n_male_spks = np.max(out_m_spk_lbs) + 1
n_male_spks = np.max(m_spk_lbs) + 1

f_x, f_spk_lbs, out_f_x, out_f_spk_lbs  = select_speakers(x_dat, sex_dat, spk_dat, channel_lbs, gender='female', min_n_vecs=30, n_spks=-1)
n_female_spks = np.max(f_spk_lbs) + 1
out_n_female_spks = np.max(out_f_spk_lbs) + 1
f_spk_lbs = [i + n_male_spks for i in f_spk_lbs]        # Speaker labels start from 0 and consecutive
out_f_spk_lbs = [i + out_n_male_spks for i in out_f_spk_lbs]


# # Load training and test data
# matfiles, gender = get_filenames2(datadir, n_gauss, n_fac)
# x_dat, sex_dat, spk_dat = load_ivectors(matfiles, gender, shuffle=False)
#
# # Select speakers with at least min_n_ivecs i-vectors (sex_trn==0 for male and ==1 for female)
# m_x, m_spk_lbs, out_m_x, out_m_spk_lbs = select_speakers(x_dat, spk_dat, sex_dat, gender='male', min_n_vecs=min_n_vecs, n_spks=-1)
# out_n_male_spks = np.max(out_m_spk_lbs) + 1
# n_male_spks = np.max(m_spk_lbs) + 1
#
# f_x, f_spk_lbs, out_f_x, out_f_spk_lbs  = select_speakers(x_dat, spk_dat, sex_dat, gender='female', min_n_vecs=min_n_vecs, n_spks=-1)
# n_female_spks = np.max(f_spk_lbs) + 1
# out_n_female_spks = np.max(out_f_spk_lbs) + 1
# f_spk_lbs = [i + n_male_spks for i in f_spk_lbs]        # Speaker labels start from 0 and consecutive
# out_f_spk_lbs = [i + out_n_male_spks for i in out_f_spk_lbs]





# Prepare training and test data (input and target)
x = np.vstack([m_x, f_x])
spk_lbs = np.hstack([m_spk_lbs, f_spk_lbs])
sex_lbs = np.asarray([0]*len(m_spk_lbs) + [1]*len(f_spk_lbs))
n_spks = np.max(spk_lbs) + 1


# trial = sio.loadmat('/home7b/lxli/matlab_mPLDA/mat/fw60/sre16_eval_tstutt_t300_w_512c.mat')
# sre16_enroll = sio.loadmat('/home7b/lxli/matlab_mPLDA/mat/fw60/sre16_eval_enrollments_t300_w_512c.mat')
all_enc = np.empty((0, 300), dtype='float32')
trial_enc = np.empty((0, 300), dtype='float32')
enroll_enc = np.empty((0, 300), dtype='float32')


# Prepare training data
x_trn = np.empty((0, x.shape[1]), dtype='float32')
spk_lbs = spk_lbs.tolist()
spk_lbs_trn = list()
sex_lbs_trn = list()
for spk in range(n_spks):
    idx = [t for t, e in enumerate(spk_lbs) if e == spk]
    idx_trn = idx
    # idx_trn = idx[0:min_n_trn_vecs]
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
# for m_spk in range(int(n_tst_spks/2)):                           # Select n_tst_spks/2 male speakers
#     idx = [t for t, e in enumerate(spk_lbs) if e == m_spk]
#     idx_tst = idx[min_n_trn_vecs:]                          # Ensure tst vectors are not the same as trn vec
#     x_tst = np.vstack([x_tst, x[idx_tst, :]])
#     spk_lbs_tst.extend([m_spk] * len(idx_tst))
#     sex_lbs_tst.extend(sex_lbs[idx_tst].tolist())
# c = int(n_tst_spks/2)
# for f_spk in range(n_male_spks, n_male_spks + int(n_tst_spks/2)): # Select n_tst_spks/2 female speakers
#     idx = [t for t, e in enumerate(spk_lbs) if e == f_spk]
#     idx_tst = idx[min_n_trn_vecs:]
#     x_tst = np.vstack([x_tst, x[idx_tst, :]])
#     spk_lbs_tst.extend([c] * len(idx_tst))
#     sex_lbs_tst.extend(sex_lbs[idx_tst].tolist())
#     c = c + 1
#
out_x = np.vstack([out_m_x, out_f_x])
out_spk_lbs = np.hstack([out_m_spk_lbs, out_f_spk_lbs])
out_sex_lbs = np.asarray([0]*len(out_m_spk_lbs) + [1]*len(out_f_spk_lbs))
out_n_spks = np.max(out_spk_lbs) + 1
out_spk_lbs = out_spk_lbs.tolist()
for m_spk in range(int(n_tst_spks/2)):                           # Select n_tst_spks/2 male speakers
# for m_spk in range(out_n_male_spks):
    idx_tst = [t for t, e in enumerate(out_spk_lbs) if e == m_spk]
    x_tst = np.vstack([x_tst, out_x[idx_tst, :]])
    spk_lbs_tst.extend([m_spk] * len(idx_tst))
    sex_lbs_tst.extend(out_sex_lbs[idx_tst].tolist())
c = int(int(n_tst_spks/2))
# c = int(out_n_male_spks)
for f_spk in range(out_n_male_spks, out_n_male_spks + int(n_tst_spks/2)): # Select n_tst_spks/2 female speakers
# for f_spk in range(out_n_male_spks, out_n_male_spks+out_n_female_spks):
    idx_tst = [t for t, e in enumerate(out_spk_lbs) if e == f_spk]
    x_tst = np.vstack([x_tst, out_x[idx_tst, :]])
    spk_lbs_tst.extend([c] * len(idx_tst))
    sex_lbs_tst.extend(out_sex_lbs[idx_tst].tolist())
    c = c + 1

spk_lbs_tst = np.asarray(spk_lbs_tst)
sex_lbs_tst = np.asarray(sex_lbs_tst)

num_hidden_D = 500
num_hidden_G = 2100
num_hidden_C = 500
dim_i_vector = 500

D_W1 = tf.Variable(xavier_init([500, num_hidden_D]))
D_b1 = tf.Variable(tf.zeros(shape=[num_hidden_D]))
D_W2 = tf.Variable(xavier_init([num_hidden_D, num_hidden_D]))
D_b2 = tf.Variable(tf.zeros(shape=[num_hidden_D]))
D_W3 = tf.Variable(xavier_init([num_hidden_D, 2]))
D_b3 = tf.Variable(tf.zeros(shape=[2]))

G_W1_D = tf.Variable(xavier_init([500, int(1*num_hidden_G/3)]))
G_W1_C = tf.Variable(xavier_init([500, int(2*num_hidden_G/3)]))
G_b1 = tf.Variable(tf.zeros(shape=[num_hidden_G]))
# G_W2_D = tf.Variable(xavier_init([num_hidden_G, int(1*num_hidden_G/3)]))
# G_W2_C = tf.Variable(xavier_init([num_hidden_G, int(2*num_hidden_G/3)]))
# G_b2 = tf.Variable(tf.zeros(shape=[num_hidden_G]))
# G_W3_D = tf.Variable(xavier_init([num_hidden_G,int(1*num_hidden_G/3)]))
# G_W3_C = tf.Variable(xavier_init([num_hidden_G, int(2*num_hidden_G/3)]))
# G_b3 = tf.Variable(tf.zeros(shape=[num_hidden_G]))
G_W4_D = tf.Variable(xavier_init([int(1*num_hidden_G/3), 500]))
G_W4_C = tf.Variable(xavier_init([int(2*num_hidden_G/3), 500]))
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
#                                           center=True, scale=True, epsilon = 0.001, decay=0.9,
#                                           is_training=phase, zero_debias_moving_mean=True)
#         return h1
#
# def generator(x, train_flag = 1):
#
#     x_norm = batch_norm_2(x, train_flag)
#
#     G_h1 = tf.matmul(x_norm, tf.concat([G_W1_C,G_W1_D],1)) + G_b1
#     G_h1_norm = batch_norm_2(G_h1, train_flag)
#     G_h1_norm = tf.nn.relu(G_h1_norm)
#
#     G_h2 = tf.matmul(G_h1_norm, tf.concat([G_W2_C,G_W2_D],1)) + G_b2
#     G_h2_norm = batch_norm_2(G_h2, train_flag)
#     G_h2_norm = tf.nn.relu(G_h2_norm)
#
#     G_h3 = tf.matmul(G_h2_norm, tf.concat([G_W3_C,G_W3_D],1)) + G_b3
#     G_h3_norm = batch_norm_2(G_h3, train_flag)
#     G_h3_norm = tf.nn.relu(G_h3_norm)
#
#     G_log_prob= tf.matmul(G_h3_norm, tf.concat([G_W4_C,G_W4_D],0)) + G_b4
#     G_prob = tf.nn.sigmoid(G_log_prob)
#
#     return G_prob
#
#
# def discriminator(z, train_flag = 1):
#     z_norm = batch_norm_2(z, train_flag)
#
#     D_h1 = tf.nn.relu(tf.matmul(z_norm, D_W1) + D_b1)
#     D_h1_norm = batch_norm_2(D_h1, train_flag)
#     D_h1_norm = tf.nn.relu(D_h1_norm)
#
#     D_h2 = tf.matmul(D_h1_norm, D_W2) + D_b2
#     D_h2_norm = batch_norm_2(D_h2, train_flag)
#     D_h2_norm = tf.nn.relu(D_h2_norm)
#
#     D_logit = tf.matmul(D_h2_norm, D_W3) + D_b3
#     D_prob = tf.nn.softmax(D_logit)
#     return D_prob, D_logit
#
# def classifier(z, train_flag = 1):
#     z_norm = batch_norm_2(z, train_flag)
#
#     C_h1 = tf.matmul(z_norm, C_W1) + C_b1
#     C_h1_norm = batch_norm_2(C_h1, train_flag)
#     C_h1_norm = tf.nn.relu(C_h1_norm)
#
#     C_h2 = tf.matmul(C_h1_norm, C_W2) + C_b2
#     C_h2_norm = batch_norm_2(C_h2, train_flag)
#     C_h2_norm = tf.nn.relu(C_h2_norm)
#
#     C_logit = tf.matmul(C_h2_norm, C_W3) + C_b3
#     C_prob = tf.nn.softmax(C_logit)
#     return C_prob, C_logit

def batch_norm(Wx_plus_b, out_size, train_flag):
        # Batch Normalize
    fc_mean, fc_var = tf.nn.moments(
        Wx_plus_b,
        axes=[0]
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

    return mean, var, shift, scale, epsilon


def generator(x, train_flag = 1):
    mean, var, shift, scale, epsilon = batch_norm(x, dim_i_vector, train_flag)
    x_norm = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

    G_h1 = tf.matmul(x_norm, tf.concat([G_W1_C,G_W1_D],1)) + G_b1
    mean, var, shift, scale, epsilon = batch_norm(G_h1, num_hidden_G, train_flag)
    G_h1_norm = tf.nn.batch_normalization(G_h1, mean, var, shift, scale, epsilon)
    G_h1_norm = tf.nn.relu(G_h1_norm)

    # G_h2= tf.matmul(G_h1_norm, tf.concat([G_W2_C,G_W2_D],1)) + G_b2
    # mean, var, shift, scale, epsilon = batch_norm(G_h2, num_hidden_G, train_flag)
    # G_h2_norm = tf.nn.batch_normalization(G_h2, mean, var, shift, scale, epsilon)
    # G_h2_norm = tf.nn.relu(G_h2_norm)
    #
    # G_h3= tf.matmul(G_h2_norm, tf.concat([G_W3_C,G_W3_D],1)) + G_b3
    # mean, var, shift, scale, epsilon = batch_norm(G_h3, num_hidden_G, train_flag)
    # G_h3_norm = tf.nn.batch_normalization(G_h3, mean, var, shift, scale, epsilon)
    # G_h3_norm = tf.nn.relu(G_h3_norm)

    G_log_prob= tf.matmul(G_h1_norm, tf.concat([G_W4_C,G_W4_D],0)) + G_b4
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob, G_log_prob


def discriminator(z, train_flag = 1):
    mean, var, shift, scale, epsilon = batch_norm(z, 500, train_flag)
    z_norm = tf.nn.batch_normalization(z, mean, var, shift, scale, epsilon)

    D_h1 = tf.matmul(z_norm, D_W1) + D_b1
    mean, var, shift, scale, epsilon = batch_norm(D_h1, num_hidden_D, train_flag)
    D_h1_norm = tf.nn.batch_normalization(D_h1, mean, var, shift, scale, epsilon)
    D_h1_norm = tf.nn.relu(D_h1_norm)

    D_h2 = tf.matmul(D_h1_norm, D_W2) + D_b2
    mean, var, shift, scale, epsilon = batch_norm(D_h2, num_hidden_D, train_flag)
    D_h2_norm = tf.nn.batch_normalization(D_h2, mean, var, shift, scale, epsilon)
    D_h2_norm = tf.nn.relu(D_h2_norm)

    D_logit = tf.matmul(D_h2_norm, D_W3) + D_b3
    D_prob = tf.nn.softmax(D_logit)
    return D_prob, D_logit

def classifier(z, train_flag = 1):
    mean, var, shift, scale, epsilon = batch_norm(z, 500, train_flag)
    z_norm = tf.nn.batch_normalization(z, mean, var, shift, scale, epsilon)

    C_h1 = tf.matmul(z_norm, C_W1) + C_b1
    mean, var, shift, scale, epsilon = batch_norm(C_h1, num_hidden_C, train_flag)
    C_h1_norm = tf.nn.batch_normalization(C_h1, mean, var, shift, scale, epsilon)
    C_h1_norm = tf.nn.relu(C_h1_norm)

    C_h2 = tf.matmul(C_h1_norm, C_W2) + C_b2
    mean, var, shift, scale, epsilon = batch_norm(C_h2, num_hidden_C, train_flag)
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

    # theta_G = [G_W1_C, G_W1_D, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]
    # theta_G_D = [G_W1_D, G_W2_D, G_W3_C, G_W4_C, G_b1, G_b2, G_b3, G_b4]
    # theta_G_C = [G_W1_C, G_W2_C, G_W3_C, G_W4_D, G_b1, G_b2, G_b3, G_b4]
    theta_G_D = [G_W1_D,  G_W4_C, G_b1, G_b4]
    theta_G_C = [G_W1_C,  G_W4_D, G_b1, G_b4]

    spk_label = tf.placeholder(tf.float32, shape=[None, spk_1h_trn.shape[1]])
    theta_C = [C_W1, C_W2, C_W3, C_b1, C_b2, C_b3]

    train_flag = tf.placeholder(tf.bool)


    #X should be i-vector, G_encoded is the target latent vector.
    G_encoded, G_log_prob= generator(X, train_flag)
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
    #
    C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_logit, labels=tf.ones_like(spk_label)/spk_1h_trn.shape[1]))

    G_loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit, labels=1 - sex_X))
    G_loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_logit, labels=spk_label))
    G_loss = G_loss_C + G_loss_D



    D_solver = tf.train.AdamOptimizer(0.001).minimize(D_loss, var_list=theta_D)
    # G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=theta_G)
    C_solver = tf.train.AdamOptimizer(0.001).minimize(C_loss, var_list=[theta_C])
    G_C_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss_C, var_list=theta_G_C)
    G_D_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss_D, var_list=theta_G_D)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')


    batch_size = 128
    n_batches = int(x_trn.shape[0] / batch_size)
    mean_data = np.mean(x_trn,axis=0)


    for it in range(20):
        pidx = np.random.permutation(x_trn.shape[0])
        for batch in range(n_batches):
            idx = pidx[batch * batch_size: (batch + 1) * batch_size]

            x = x_trn[idx]
            y_sex_1h = sex_1h_trn[idx]
            y_spk_1h = spk_1h_trn[idx]


            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x, sex_X: y_sex_1h, train_flag: 1})

            _, G_loss_D_curr = sess.run([G_D_solver, G_loss_D], feed_dict={X: x, sex_X:y_sex_1h, spk_label: y_spk_1h,
                                                                           train_flag: 1})

            # if G_loss_D_curr > 2:
            #     for i in range(int(G_loss_D_curr)):
            #         _, G_loss_D_curr = sess.run([G_D_solver, G_loss_D], feed_dict={X: x, sex_X:
            #                                                                 y_sex_1h,spk_label: y_spk_1h,train_flag: 1})
            # _, G_loss_C_curr = sess.run([G_C_solver, G_loss_C], feed_dict={X: x, sex_X: y_sex_1h, spk_label: y_spk_1h,
            #                                                                train_flag: 1})
            for i in range(4):
                _, G_loss_C_curr = sess.run([G_C_solver, G_loss_C], feed_dict={X: x, sex_X: y_sex_1h, spk_label: y_spk_1h,
                                                                               train_flag: 1})
            if it % 3 == 0:
                _, C_loss_curr = sess.run([C_solver, C_loss], feed_dict={X: x, spk_label: y_spk_1h, train_flag: 1})
            # for i in range(2):


            if it % 1 == 0 and batch == 1:
                print('Iter: {}'.format(it))
                # print('D loss: {:.4}'. format(D_loss_curr))
                # print('G_loss: {:.4}'.format(G_loss_curr))
                print('G_loss_C: {:.4}'.format(G_loss_C_curr))
                # print('G_loss_D: {:.4}'.format(G_loss_D_curr))
                print('C_loss: {:.4}'.format(C_loss_curr))
                # print('D_solver: {:.4}'.format(D_logit_fake_curr))
                print()
            # fig, _, _, _ lock=False)
            if it % 1  == 0 and batch == 0 :
                # mean, var, shift, scale, epsilon = batch_norm(x_tst,  tf.float32(500))
                # x_tst_norm = tf.nn.batch_normalization(x_tst, mean, var, shift, scale, epsilon)
                x_tst_enc = sess.run(G_log_prob, feed_dict={X: x_tst, train_flag: 1})
                print('Creating t-SNE plot')
                x_tst_enc_prj = TSNE(random_state=20150101).fit_transform(x_tst_enc)
                # fig, _, _, _ = scatter2D(x_tst_enc_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
                fig, _, _, _ = scatter2D(x_tst_enc_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
                                             title='Adversarially Transformed I-Vectors (Epoch = %d)' % it)
                filename = logdir + 'aae4-%d-%d-epoch%d.png' % (500, min_n_vecs, it)
                fig.savefig(filename)
                plt.show(block=False)

    print('Creating t-SNE plot')
    x_prj = TSNE(random_state=20150101).fit_transform(x_tst)
    fig, _, _, _ = scatter2D(x_prj, spk_lbs_tst, markers=sex_lbs_tst, n_colors=n_tst_spks,
                             title='Original i-vectors')
    filename = logdir + 'ivec-%d.png' % n_fac
    fig.savefig(filename)
    plt.show()

    # all_enc = sess.run(G_encoded, feed_dict={X: x_dat, train_flag: 1})
    # sio.savemat('data/encoded_preSRE.mat', {'w': all_enc})
    # trial_enc = sess.run(G_encoded, feed_dict={X: trial['w'], train_flag: 1})
    # sio.savemat('data/encoded_tst_SRE16.mat',
    #             {'w': trial_enc, 'spk_logical': trial['spk_logical'], 'spk_physical': trial['spk_physical'],
    #              'num_frames': trial['num_frames']})
    # enroll_enc = sess.run(G_encoded, feed_dict={X: sre16_enroll['w'], train_flag: 1})
    # sio.savemat('data/encoded_enroll_SRE16.mat', {'w': enroll_enc, 'spk_logical': sre16_enroll['spk_logical'], 'spk_physical': trial['spk_physical'],
    #              'num_frames': trial['num_frames']})
    sre12_trn_enc = sess.run(G_encoded, feed_dict={X: x_trn, train_flag: 1})
    sio.savemat('data/sre12_trn_enc.mat',
                {'w': sre12_trn_enc, 'spk_logical': spk_lbs_trn})
    sio.savemat('data/sre12_trn.mat',
                {'w': x_trn, 'spk_logical': spk_lbs_trn})

    sre12_tst_enc = sess.run(G_log_prob, feed_dict={X: x_tst, train_flag: 1})
    sio.savemat('data/sre12_tst_enc.mat',
                {'w': sre12_tst_enc, 'spk_logical': spk_lbs_tst})
    sio.savemat('data/sre12_tst.mat',
                {'w': x_tst, 'spk_logical': spk_lbs_tst})
    print("Successfully Saved Mat File.")

if __name__ == '__main__':
    # Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    np.random.seed(1)

    main()

