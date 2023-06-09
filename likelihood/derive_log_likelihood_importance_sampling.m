%% This function returns the log likelihood of test data across all stimuli and test trials, given the covariates

% Inputs:   ysamp_test: spike counts of test data (size = time bins * test trials * stimuli) 
%           Signal_latents: signal latents estimated from trainig data (size = time bins  * stimuli) 
%           dtbin: the bin size in seconds
%           xsamp: the time axis in seconds (size = time bins * 1) 
%           Noise_mean_all: mean of noise variational distribution on test data (size = time bins * test trials * stimuli) 
%           Noise_coviarance_all: covariance of noise variational distribution on test data (size = time bins * time bins * test trials * stimuli) 
%           estimated_hyperparameters: the final hyper-parameter estimates
%           MC_interations: number of Monte Carlo iterations for Importance Sampling

% Output:   log_likelihood_across_trial_stimuli: the log likelihood across all trials and stimuli

function [log_likelihood_across_trial_stimuli,weights] = derive_log_likelihood_importance_sampling(ysamp_test,Signal_latents,dtbin,xsamp,Noise_mean_all,Noise_coviarance_all,estimated_hyperparameters,MC_interations,Matern_order)

% Initialize the variables
log_likelihood_across_trial_stimuli = 0;
P = size(ysamp_test,3); % number of stimuli
L_test = size(ysamp_test,2); % number of test trials
weights = 0;

additive_constant = 10^(-2); % adding a small constant to noise covariances in case it is very small, which would lead for the inverse to be Inf

% This finds the log likelihood of the baseline Poisson model with no noise
if nargin < 4 
    
    for p = 1:P
        for l = 1:L_test
            ysamp = squeeze(squeeze(ysamp_test(:,l,p))); % spike counts
            signal = squeeze(Signal_latents(:,p)); % the signal latent estimate
            noise = signal*0; % no noise
            [~,log_likelihood_temp] = Poisson_conditional_likelihood(ysamp,signal,noise,dtbin); % derive the conditional Poisson log likelihood
            log_likelihood_across_trial_stimuli = log_likelihood_across_trial_stimuli + log_likelihood_temp;
        end
    end
    
% For the other models, derive the log likelihood using Importance Sampling
else
    weights = zeros(MC_interations,L_test,P);
    
    for p = 1:P
        for l = 1:L_test
            ysamp = squeeze(squeeze(ysamp_test(:,l,p))); % spike counts
            signal = squeeze(Signal_latents(:,p)); % the signal latent estimate
            if size(estimated_hyperparameters,2) > 1 % Independent noise model
                hyperparameters = estimated_hyperparameters{p};
            else    % Shared noise model
                hyperparameters = estimated_hyperparameters;
            end
            if nargin > 8
                noise_cov_prior = get_Matern_cov(xsamp,hyperparameters.noise.len,hyperparameters.noise.rho,Matern_order);
            else
                noise_cov_prior = get_GP_cov_from_hyp(hyperparameters,xsamp); % get the noise covariance prior from hyper-parameters
            end
            noise_mean_var_dist = squeeze(squeeze(Noise_mean_all(:,l,p))); % mean of noise distribution
            noise_cov_var_dist = squeeze(squeeze(Noise_coviarance_all(:,:,l,p))); % covariance of noise distribution      
            noise_cov_var_dist = (noise_cov_var_dist + noise_cov_var_dist')/2; % ensure symmetry incase of numerical errors
            log_likelihood_temp = zeros(1,MC_interations); % store the LL for all MC iterations
            % The Monte Carlo iterations
            for i = 1:MC_interations
                % Derive a noise sample from the Variational Distribution
                noise = draw_noise_sample_var_dis(noise_mean_var_dist,noise_cov_var_dist+additive_constant*eye(size(noise_cov_var_dist)));
                % Conditional Poisson data likelihood
                [~,data_log_likelihood] = Poisson_conditional_likelihood(ysamp,signal,noise,dtbin);
                % Variational Distribution likelihood
                var_dis_log_likelihood = Gaussian_log_likelihood(noise, noise_mean_var_dist, noise_cov_var_dist+additive_constant*eye(size(noise_cov_var_dist)));
                % Prior likelihood
                prior_log_likelihood =  Gaussian_log_likelihood(noise, noise*0, noise_cov_prior+additive_constant*eye(size(noise_cov_var_dist)));
                % Combining the three components
                log_likelihood_temp(1,i) = data_log_likelihood + prior_log_likelihood - var_dis_log_likelihood;
%                 weights(i,l,p) = prior_log_likelihood - var_dis_log_likelihood;
            end
            % Averaging across iterations using the log-sum-exp trick
            log_likelihood_across_trial_stimuli = log_likelihood_across_trial_stimuli + logsumexp(log_likelihood_temp,2) - log(MC_interations);
        end
    end
    
end