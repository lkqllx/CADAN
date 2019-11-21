function scores = Cos_GroupScoring(W1, w2)
n_tgt_utts = size(W1,2);
scores = zeros(n_tgt_utts,1);
for i = 1:n_tgt_utts,
    w1 = W1(:, i);
    scores(i) = cosine_dist(w1,w2);
end