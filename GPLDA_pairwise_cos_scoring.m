function scoremat = GPLDA_pairwise_cos_scoring(tst_w, trn_w)
% Return a pairwise score matrix of a set of i-vectors (row vectors)

% Set up path to use GPLDA package
if (strcmp(computer,'PCWIN')==1 || strcmp(computer,'PCWIN64')==1),
    addpath 'D:/so/Matlab/PLDA/BayesPLDA';  
else
    addpath '~/so/Matlab/PLDA/BayesPLDA';
end   

n_trn = size(trn_w,1);
n_tst = size(tst_w,1);
scoremat = zeros(n_tst,n_trn);

% trn_w = (preprocess_ivecs(gplda, trn_w'))';
% tst_w = (preprocess_ivecs(gplda, tst_w'))';
% for i=1:n_tst,
%     fprintf('Scoring utt %d of %d\r',i,n_tst);
% %     matlabpool open;
%     for j=1:n_trn,
D = pdist(tst_w', 'cosine');
scoremat = squareform(D);
for i= 1:size(scoremat),
    scoremat(i,i) = 1;
end

%         scoremat(j,i) = scoremat(i,j);
%     end
%     matlabpool close;
    
end