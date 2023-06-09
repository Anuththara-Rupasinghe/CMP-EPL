%% This function evaluates the conditional Poisson data likelikood given the underlying covariates

% Inputs:   ysamp: spike counts (size = time bins * 1) 
%           signal: signal latents (size = time bins  * 1) 
%           noise: noise latents (size = time bins  * 1) 
%           dtbin: the bin size in seconds

% Output:   conditional_likelihood: the likelihood based on the Poisson model
%           conditional_log_likelihood: the log likelihood based on the Poisson model

function [conditional_likelihood, conditional_log_likelihood] = Poisson_conditional_likelihood(ysamp,signal,noise,dtbin)
    
    % Initialize variables
    conditional_likelihood = 1;
    conditional_log_likelihood = 0;
    
    % Combine the likelihood across all time samples
    for t = 1:length(ysamp)
        lambda = dtbin*exp(signal(t) + noise(t));
        ysamp_t  = ysamp(t);
        if ysamp_t < 1
            ysamp_t = 0;
        end
        conditional_likelihood = conditional_likelihood *(exp(-1*lambda)*(lambda^(ysamp_t))/(factorial(ysamp_t)));
        conditional_log_likelihood = conditional_log_likelihood - lambda + ysamp_t.*log(lambda) - log(factorial(ysamp_t));
    end

end