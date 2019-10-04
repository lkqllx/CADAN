% Obtain speaker identifiation accuracy

sre12_enc = load ('data/gmix_t300_gplda_512c_sre12_enc.mat');
sre12_enc = sre12_enc.GPLDAModel;
sre12 = load ('data/gmix_t300_gplda_512c_sre12.mat');
sre12 = sre12.GPLDAModel;


sre12_tst_enc = load ('data/sre12_tst_enc.mat');
sre12_tst_enc.w = double(sre12_tst_enc.w);
sre12_tst_enc.spk_logical = sre12_tst_enc.spk_logical';
sre12_tst = load ('data/sre12_tst.mat');
sre12_tst.w = double(sre12_tst.w);
sre12_tst.spk_logical = sre12_tst.spk_logical';


% sre12_scoremat_enc = GPLDA_pairwise_scoring(sre12_enc,sre12_tst_enc.w);
% sre12_scoremat = GPLDA_pairwise_scoring(sre12,sre12_tst.w);
% sre12_scoremat_ori_enc = GPLDA_pairwise_scoring(sre12,sre12_tst_enc.w);

sre12_acc_enc = get_spkid_acc(sre12_scoremat_enc,sre12_tst_enc.spk_logical);
sre12_acc = get_spkid_acc(sre12_scoremat,sre12_tst.spk_logical);
















