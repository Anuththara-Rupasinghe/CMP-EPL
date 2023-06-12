%% This script performs a unittest to test that the function derive_ELBO_VI_CMP_EPL is functiioning as expected

clear all;
close all;
clc;

%% Step 1: Specify the ground truth hyper-parameters, and simulate the latent signal, noise activity and discretize spike counts

T_max = 25.6; % total duration of each trial in s
dtbin = 0.2; % specify the desired bin size for discretization
L = 30; % specify the number of trials available for inference

% Specify the true Signal Hyper-parameters
true_hyperparameters.signal.rho = 2;
true_hyperparameters.signal.len = 2;

% Specify the true Noise Hyper-parameters
true_hyperparameters.noise.rho = 0.8;
true_hyperparameters.noise.len = 0.3;
true_hyperparameters.noise.q = 1;

% Exponential nonlinearity
nlfun = @myexp; % exponential nonlinearity
    
% Derive the spike trains by simulations
[ysamp,xsamp,ztrue_noise,ztrue_signal,ftrue] = simulate_spike_observations(true_hyperparameters,T_max,dtbin,L,nlfun);

%% Step 2: Infer the latent signal, noise activity based on true hyper-parameters and spike counts using the derive_ELBO_VI_CMP_EPL.m function

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_q_final = [true_hyperparameters.noise.rho,true_hyperparameters.noise.len,true_hyperparameters.noise.q];
signal_rho_len_final = [true_hyperparameters.signal.rho, true_hyperparameters.signal.len];

% Evaluate the final latent estimates
[~,Noise_latents,Signal_latents,Firing_rate_per_bin] = derive_ELBO_VI_CMP_EPL(signal_rho_len_final,noise_rho_len_q_final,xsamp,ysamp,dtbin,tolerance_lambda_final,use_log);

%% Step 3: Compare the ground truth and inferred signal/noise latent variables using pairwise correlations, and return true if they are positively correlated

temp = corrcoef(Firing_rate_per_bin/dtbin,ftrue);
cor_firing_rate = temp(1,2);
temp = corrcoef(ztrue_noise,Noise_latents);
cor_noise = temp(1,2);
temp = corrcoef(ztrue_signal,Signal_latents);
cor_signal = temp(1,2);

cor_threshold = 0.4;

if (cor_firing_rate > cor_threshold) && (cor_noise > cor_threshold) && (cor_signal > cor_threshold)
 fprintf('unit test passesd!!! \n')
else
 fprintf('unit test failed: check derive_ELBO_VI_CMP_EPL function!!! \n')
end