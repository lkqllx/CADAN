% Compute the cosine-distance scores, given a set of target speaker i-vec and test i-vec
%
% Input:
%   target_w_file        - Cell array containing n i-vectors files 
%   tnorm_w_file         - Remain here for backward compatibility. Should be empty
%   test_w_file          - File containing test i-vectors
%   ndx_lstfile          - List file specifying the evaluation trials
%   znorm_para_file      - Remain here for backward compatibility. Should be empty
%   ztnorm_para_file     - Remain here for backward compatibility. Should be empty
%   normType             - Remain here for backward compatibility. Should be 'None'
%   evlfile              - Output file in NIST SRE format
%   opt                  - Optional parameters controlling the behaviour of the scoring process
% Example:
%   scripts/comperr.pl -evl evl/cds60_mixgender_test_256c.evl -sscr scr/cds60_mixgender_test_256c_spk.scr -iscr scr/cds60_mixgender_test_256c_imp.scr
%   No. of trials         : 761208 
%   No. of target trials  : 5516 
%   No. of impostor trials: 755692 
%   Ts=5516, Tb=755692
%   ActTh=1.000000  FAR=0.000000    FRR=0.000000
%   EERth=0.061890  EER=1.123606
%   minDCFth=0.084329  FAR=0.160250  FRR=3.226976  MinDcf=0.004813

%   % Author: M.W. Mak
% Date: 7 June 2015
%
function scores = score_cds_w(target_w_file,tnorm_w_file,test_w_file,ndx_lstfile,...
                                znorm_para_file, ztnorm_para_file, normType,...
                                evlfile, opt)
                            
% opt.mode = 'scravg': Scoring test ivec against target ivecs individually (default)
% opt.mode = 'ivcavg': Scoring test ivec against the averaged i-vecs within each SNR group without average the SNR
if ~exist('opt','var'),
    opt.mode = 'scravg';          
end 
                            
if nargin ~= 8 && nargin ~=9,
    disp('Need 8 or 9 input arguments');
    return;
end

% Load i-vectors of target speakers.
% Concatenate the i-vectors, spk_logical, spk_physical, num_frames in the input target_w_files
trn = cell(length(target_w_file)); 
tgt = struct('w',[],'spk_logical',[],'spk_physical',[],'num_frames',[]);
for i=1:length(target_w_file),
    disp(['Loading ' target_w_file{i}]);
    trn{i} = load(target_w_file{i});       % Load w and spk_logical of target speakers
    tgt.w = [tgt.w; trn{i}.w];
    tgt.spk_logical = [tgt.spk_logical; trn{i}.spk_logical];
    tgt.spk_physical = [tgt.spk_physical; trn{i}.spk_physical];
    tgt.num_frames = [tgt.num_frames; trn{i}.num_frames];
end
clear trn;

% Load the test i-vectors
disp(['Loading ' test_w_file]);
tst = load(test_w_file);

% Load NORM models (w)
switch (normType)
    case 'Znorm'
        disp(['Loading ' znorm_para_file]);
        normp.znm = load(znorm_para_file);
    case {'Tnorm'}
        disp(['Loading ' tnorm_w_file]);
        normp.tnm = load(tnorm_w_file);
        normp.tnm_w = normp.tnm.w;
    case {'ZTnorm1'}
        disp(['Loading ' znorm_para_file]);
        normp.znm = load(znorm_para_file);        
        disp(['Loading ' tnorm_w_file]);
        normp.tnm = load(tnorm_w_file);
        normp.tnm_w = normp.tnm.w;
    case {'ZTnorm2'}
        disp(['Loading ' znorm_para_file]);
        normp.znm = load(znorm_para_file);        
        disp(['Loading ' tnorm_w_file]);
        normp.tnm = load(tnorm_w_file);
        normp.tnm_w = normp.tnm.w;
        disp(['Loading ' ztnorm_para_file]);
        normp.ztnm = load(ztnorm_para_file);
    case {'None'}
        disp('No norm');
        normp = struct('tnm',[],'znm',[],'tnm_w',[],'ztnm',[]);
    otherwise
        disp('Incorrect norm type');
        return;
end


% Compute the score of all trials, calling cosine_dist() one trial at a time.
mode = opt.mode;
fprintf('Scoring mode: %s\n', mode);
num_tests = numlines(ndx_lstfile);
scores = zeros(num_tests,1);
ndx.spk_logical = parse_list(ndx_lstfile);
n_tstutt = length(tst.spk_logical);
C_target = cell(n_tstutt,1);
C_testutt = cell(n_tstutt,1);
C_channel = cell(n_tstutt,1);

% To speed up, avoid using indexed array
ndx_spk_logical = ndx.spk_logical;
tgt_spk_logical = tgt.spk_logical;

for i=1:num_tests,                           % For each trial
    session_name = ndx_spk_logical{i};       % Test session name
    field = rsplit(':',session_name);        % Split spk_logical into target and test utt (e.g. 100396:tabfsa_sre12_B)
    target = field{1};
    testutt = field{2};
    field = rsplit('_',testutt);
    channel = lower(field{end});
        
    % Find the index of the test utt. Note: use spk_physical as test utt ID
    k = find(strncmp(testutt, tst.spk_physical,length(testutt))==1); 
    tst_w = tst.w(k,:);                              % k should be a scalar 
   
    % Find target session of the current target speaker
    tgt_sessions = find(strncmp(target, tgt_spk_logical,length(target))==1);  % Find out against which target the utt is tested

    % Make sure that target sessions exist
    assert(~isempty(tgt_sessions),sprintf('Missing sessions of %s in score_cds_w.m',target));    
    
    % Get i-vectors of target's training sessions. 
    tgt_w = tgt.w(tgt_sessions,:);
        
    % Compute the scores of current tst utt against all training utts of the selected target speaker
    % Reject this speaker if the speaker does not have enrollment utterances
    if (isempty(tgt_sessions)~=1),
        switch(mode)
            case {'scravg'}
                cds_scr = cosine_dist(tgt_w', tst_w');
            case {'ivcavg'}
                tgt_w = mean(tgt_w,1);
                cds_scr = cosine_dist(tgt_w', tst_w');
            otherwise
                disp('Invalid opt.mode parameter');
        end
    else
        cds_scr = -174; % Should not reach here if the line assert() above is in effect.
    end
    
    % Perform score normalization (if necessary) and compute the mean CDS score
    if (strcmp(normType,'None')==1),
        scores(i) = mean(cds_scr);
    else
        tgt_num = find(strcmp(target, normp.znm.spk_id)==1);            % Find the target number (index in normp)
        scores(i) = mean(normalization(cds_scr, tst_w, testutt, normType, normp, tgt_num));
    end
    
    % Show scoring progress
    if mod(i-1,10000)==0,
        fprintf('(%d/%d) %s,%s: %f\n',i,num_tests,target,testutt,scores(i));
    end
    
    % Copy target, testutt and channel to cell array for saving to file later
    C_target{i} = target;
    C_testutt{i} = testutt;
    C_channel{i} = channel;
       
end

% Save the score to .evl file
fp = fopen(evlfile,'w');
for i=1:num_tests,
    fprintf(fp,'%s,%s,%c,%.7f\n', C_target{i}, C_testutt{i}, 'a', scores(i));    
end
fclose(fp);

disp(['Scores saved to ' evlfile]);

return;


%% private function




function scr = normalization(cds_scr, tst_w, testutt, normType, normp, tgt_num)
% Create a hash table containing arrays of doubles [mu sigma] that have
% been found in previous test sessions.
% Use the session name as the key. <session_name,[mu sigma]>global sessionhash;
sessionhash = java.util.Hashtable;
switch(normType)
    case {'Znorm'}
        scr = (cds_scr - normp.znm.mu(tgt_num))./normp.znm.sigma(tgt_num);    
    case {'Tnorm'}
        if (sessionhash.containsKey(testutt) == 0),
            [mu,sigma] = comp_tnorm_para(tst_w, normp.tnm_w);
            sessionhash.put(testutt,[mu sigma]);
        else
            tnmpara = sessionhash.get(testutt); % Java methods can return array of double
            mu = tnmpara(1); 
            sigma = tnmpara(2);
        end
        scr = (cds_scr - mu)/sigma;
    case {'ZTnorm1'}
        cds_scr = (cds_scr - normp.znm.mu(tgt_num))./normp.znm.sigma(tgt_num);    
        if (sessionhash.containsKey(testutt) == 0),
            [mu,sigma] = comp_tnorm_para(tst_w, normp.tnm_w);
            sessionhash.put(testutt,[mu sigma]);
        else
            tnmpara = sessionhash.get(testutt); % Java methods can return array of double
            mu = tnmpara(1); 
            sigma = tnmpara(2);
        end
        scr = (cds_scr - mu)/sigma;
    case {'ZTnorm2'}
        cds_scr = (cds_scr - normp.znm.mu(tgt_num))./normp.znm.sigma(tgt_num);    
        if (sessionhash.containsKey(testutt) == 0),
            [mu,sigma] = comp_ztnorm_para(tst_w, normp.tnm_w, normp.ztnm);
            sessionhash.put(testutt,[mu sigma]);
        else
            ztnmpara = sessionhash.get(testutt); % Java methods can return array of double
            mu = ztnmpara(1); 
            sigma = ztnmpara(2);
        end
        scr = (cds_scr - mu)/sigma;
    case {'None'}
        scr = cds_scr;
end



function [mu,sigma] = comp_tnorm_para(tst_w, tnm_w)
N = size(tnm_w,1);
scores = zeros(N,1);
for i=1:N,
    scores(i) = cosine_dist(tst_w',tnm_w(i,:)');
end
mu = mean(scores);
sigma = std(scores);


function [mu,sigma] = comp_ztnorm_para(tst_w, tnm_w, ztnm)
N = size(tnm_w,1);
scores_z = zeros(N,1);
for k=1:N,
    scores_z(k) = (cosine_dist(tst_w',tnm_w(k,:)')-ztnm.mu(k))/ztnm.sigma(k);
end
mu = mean(scores_z);
sigma = std(scores_z);

% Return the cosine distance between X and y. If X is a column vectors, cds is also a col vec.
function cds = cosine_dist(X, y) 
cds = zeros(size(X,2),1);
for i=1:size(X,2),
    x = X(:,i);
    cds(i) = x'*y / (norm(x)*norm(y));
end


