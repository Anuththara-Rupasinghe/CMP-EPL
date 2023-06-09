%% This function derives the negative log-likihood of spike counts given the hyper-parameters for the Goris independent model

% Inputs:   r: shape parameter of the Gamma distribution
%           s_all: scale parameters of the Gamma distributions at different time points (size: time bins * 1)
%           ysamp:  the spike counts (size: time bins * trials)

% Outputs:  neg_log_like: The total negative log-likelihood across all trials

function neg_log_like = derive_neg_log_like_Goris_independent_model(r,s_all,ysamp)
    
    s_all = s_all(:);
    K = length(s_all);
    neg_log_like = 0;
    for k = 1:K
        neg_log_like = neg_log_like + derive_neg_log_like_baseline_Goris_model(r,s_all(k),squeeze(ysamp(k,:)));
    end
    
end