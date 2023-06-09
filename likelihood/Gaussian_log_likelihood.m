%% This function evaluates the Gaussian log likelihood of a sample given the mean and covariance

% Inputs:   sample: a single realization (size = time bins * 1) 
%           mean: mean of the distribution (size = time bins  * 1) 
%           cov: covariance of the distribution (size = time bins  * time bins) 

% Output:   log_likelihood: the Gaussian log likelihood

function log_likelihood = Gaussian_log_likelihood(sample, mean, cov)

    K = length(sample);
    sample = reshape(sample,[K,1]);
    mean = reshape(mean,[K,1]);
    log_likelihood = -0.5*(K*log(2*pi) + log(det(cov)) + (sample - mean)' * (pinv(cov)) * (sample - mean));

end