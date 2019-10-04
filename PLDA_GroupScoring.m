%=========================================================
% function scores = PLDA_GroupScoring(PLDAModel, W1, w2)
% Perform PLDA scoring based on the paper
% Analysis of I-vector Length Normalization in Speaker Recognition Systems
% Note: The i-vecs should have been pre-processed by preprocess_ivecs.m
%   Input:
%       PLDAModel      - GPLDA model structure
%       W1             - Matrix containing a set of i-vectors in column
%       w2             - Second un-normalized i-vector (column vec)
%   Output:
%       scores         - PLDA scores (unnormalized) of W1 and w2
%       
% Author: M.W. Mak
% Date: July 2012
%=========================================================
function scores = PLDA_GroupScoring(PLDAModel, W1, w2)

P = PLDAModel.P;
Q = PLDAModel.Q;
n_tgt_utts = size(W1,2);
scores = zeros(n_tgt_utts,1);
const = PLDAModel.const + 0.5*w2'*Q*w2;
Pw2 = P*w2;
for i = 1:n_tgt_utts,
    w1 = W1(:, i);
    scores(i) = const + 0.5*w1'*Q*w1 + w1'*Pw2;
end

% Short-hand form but contains extra computation
% scores = PLDAModel.const + 0.5*(diag(W1'*Q*W1) + w2'*Q*w2 + 2*W1'*P*w2);
