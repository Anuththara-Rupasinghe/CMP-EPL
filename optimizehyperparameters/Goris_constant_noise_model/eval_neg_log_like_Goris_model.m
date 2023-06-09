%% This function derives the negative log-likihood of spike counts given the hyper-parameters for the Goris model (Goris 2014)

% Inputs:   hyp: a vector containing the [shape, scale] parameter of the Gamma distribution (size: 1 * 2)
%           spike_count_trials:  the spike counts (size: 1 * trials)

% Outputs:  neg_log_like: The total negative log-likelihood across all trials

function neg_log_like = eval_neg_log_like_Goris_model(hyp,spike_count_trials)

    r = hyp(1); s = hyp(2);
    neg_log_like = 0;
    for l = 1:length(spike_count_trials)
        N = spike_count_trials(l);
        neg_log_like = neg_log_like -(log(gamma(N+r)) - log(gamma(N+1)) - log(gamma(r)) + N*log(s) - (N+r)*log(1+s));
    end
    
end