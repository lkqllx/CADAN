function scoremat = GPLDA_pairwise_scoring(gplda, tst_w, trn_w)
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

trn_w = (preprocess_ivecs(gplda, trn_w'))';
tst_w = (preprocess_ivecs(gplda, tst_w'))';
for i=1:n_tst,
    fprintf('Scoring utt %d of %d\r',i,n_tst);
%     matlabpool open;
    for j=1:n_trn,
        scoremat(i,j) = PLDA_GroupScoring(gplda, tst_w(i,:)', trn_w(j,:)');
        scoremat(j,i) = scoremat(i,j);
    end
%     matlabpool close;
    
end

function W = preprocess_ivecs(PLDAModel, W)
n_vec = size(W,2);
W = PLDAModel.projmat1' * (W-repmat(PLDAModel.meanVec1,1,n_vec));    % WCCN Whitening
W = (len_norm(W'))';                                                 % Length norm
W = PLDAModel.projmat2' * W;                                         % LDA+WCCN   

% Subtract the global mean of pre-processed training i-vectors (after WCCN+lennorm+LDA+WCCN)
% from the current set of i-vectors so that the scoring function (using P & Q) 
% becomes independent on the global mean
% W = W-repmat(PLDAModel.meanVec2, 1, n_vec);
return;