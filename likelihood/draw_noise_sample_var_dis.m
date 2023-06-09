%% This function outputs a sample from a Gaussian Distribution

% Inputs:    mean: mean of the distribution (size = time bins  * 1) 
%            covariance: covariance of the distribution (size = time bins * time bins) 

% Output:   noise_simulated: a simulated sample from the Gaussian Distribution

function noise_simulated = draw_noise_sample_var_dis(mean,covariance)
 
    noise_simulated = mvnrnd(mean,covariance);

end