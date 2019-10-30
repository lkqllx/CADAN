import scipy.io as sio

mat = sio.loadmat('/Users/andrew/Desktop/HKUST/Projects/Self/CADAN/alt_data/male_target-mic-06dB_mix_t500_w_1024c.mat')
print(mat.w)