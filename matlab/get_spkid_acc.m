function acc = get_spkid_acc(scoremat, spk_logical, trn_spk_logical)
% Return speaker ID accuracy based on a pairwise score matrix and spkID info

[~,~,spkid] = unique(spk_logical);
n_spks = max(spkid);
n_ivecs = length(spk_logical);

% Find the session index of each test speaker
sessions = cell(n_spks,1);
for s = 1:n_spks,
   sessions{s} = find(spkid == s); 
end

% For each test i-vectors, find the average PLDA scores of individual speakers
scores = zeros(n_ivecs, n_spks);
maxpos = zeros(n_ivecs, 1);
for i = 1:n_ivecs,
    for s = 1:n_spks,
        sess = setdiff(sessions{s},i);                  % Do not count self-comparison
        scores(i,s) = mean(scoremat(i,sess));
    end
    [~,maxpos(i)] = max(scores(i,:));
end

% For each test i-vectors, if the maxpos matches the spkid, the i-vec is correctly classified;
n_correct = 0;
for i = 1:n_ivecs,
    if maxpos(i) == spkid(i),
        n_correct = n_correct + 1;
    end
end
acc = n_correct/n_ivecs;




% load 'data/sre12_tst.mat'
% X = pdist(double(w),'cosine');
% M = squareform(X);
% load 'data/sre12_tst_enc.mat'
% X_enc = pdist(double(w),'cosine');
% M_enc = squareform(X_enc);
% acc = get_spkid_acc(M,spk_logical');
% acc_enc = get_spkid_acc(M_enc,spk_logical');

% addpath (genpath('../../matlab_mPLDA/mixPLDA'));
% load 'data/sre12_tst.mat'
% w_tst = double(w);
% spk_logical_tst = spk_logical;
% load 'data/sre12_tst_enc.mat'
% w_tst_enc = double(w);
% load 'data/sre12_trn.mat'
% w_trn = double(w);
% spk_logical_trn = spk_logical';
% load 'data/sre12_trn_enc.mat'
% w_trn_enc = double(w);
% load 'data/sre12_all_male.mat'
% w_male = double(w);
% spk_logical_male = spk_logical';
% load 'data/sre12_all_female.mat'
% w_female = double(w);
% spk_logical_female = spk_logical';
% load 'data/sre12_all_male_enc.mat'
% w_male_enc = double(w);
% load 'data/sre12_all_female_enc.mat'
% w_female_enc = double(w);
% 
% 
% GPLDAModel=comp_GPLDA1(w_female, spk_logical_female, 'data/gmix_t300_gplda_512c_sre12.mat', 200, 200);
% scoremat = GPLDA_pairwise_scoring(GPLDAModel,w_tst, w_tst);
% 
% acc = get_spkid_acc(scoremat,spk_logical_tst ,spk_logical_male);
% 
% 
% 
% addpath (genpath('../../matlab_mPLDA/mixPLDA'));
% load 'data/sre_all_male.mat'
% w_male = double(w);
% spk_logical_male = spk_logical';
% load 'data/sre_all_female.mat'
% w_female = double(w);
% spk_logical_female = spk_logical';
% load 'data/sre_all_male_enc.mat'
% w_male_enc = double(w);
% load 'data/sre_all_female_enc.mat'
% w_female_enc = double(w);
% 
% GPLDAModel=comp_GPLDA1(w_female(1:1000,:), spk_logical_female(1:1000,:), 'data/gmix_t300_gplda_512c_sre12.mat', 200, 200);
% scoremat = GPLDA_pairwise_scoring(GPLDAModel,w_female(1001:1432,:), w_female(1001:1432,:));
% 
% acc = get_spkid_acc(scoremat,spk_logical_female(1001:1432,:) ,spk_logical_male);





