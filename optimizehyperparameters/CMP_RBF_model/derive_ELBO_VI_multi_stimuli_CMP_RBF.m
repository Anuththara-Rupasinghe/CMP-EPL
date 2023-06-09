%% This function computes the ELBO given hyper-parameters for the RBF noise kernel model (CMP-RBF) for all stimuli

% Inputs:   signal_rho_len: a vector of length 2*stimuli containing [signal rho, signal length] for each stimuli
%           noise_rho_len: a vector of length 2 containing [noise rho, noise length]
%           ysamp:  the spike counts (size: time bins * trials * stimuli)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts
%           tolerance_lambda: the threshold for checking convergence of estimated firing rate
%           use_log: a logical variable indicating whether the inference is performed in log-domain

% Outputs:  ELBO: Derived Evidence Lower BOund
%           B_mu_all: Estimated noise latents (size: time bins * trials * stimuli)
%           B_nu: Estimated signal latents (size: time bins * stimuli)
%           lambda_hat_all: Estimated firing rates per bin (size: time bins * trials * stimuli)
%           B_Sigma_B_all: Estimated noise covariances in time domain (size: time bins * time bins * trials * stimuli)

function [ELBO,B_mu_all,B_nu_all,lambda_hat_all,B_Sigma_B_all] = derive_ELBO_VI_multi_stimuli_CMP_RBF(signal_rho_len,noise_rho_len,xsamp,ysamp,dtbin,tolerance_lambda,use_log)  
    
    ELBO = 0;
    B_mu_all = ysamp*0;
    lambda_hat_all = B_mu_all;
    B_nu_all = zeros(size(ysamp,1),size(ysamp,3));
    B_Sigma_B_all = zeros(size(ysamp,1), size(ysamp,1), size(ysamp,2), size(ysamp,3));
    
    J = size(ysamp,3); % number of different stimuli
    
    % Derive the covariates for each stimulus
    for j = 1:J
        [ELBO_temp,B_mu,B_nu,lambda_hat,B_Sigma_B] = derive_ELBO_VI_CMP_RBF(squeeze(signal_rho_len(j,:)),noise_rho_len,xsamp,squeeze(ysamp(:,:,j)),dtbin,tolerance_lambda,use_log);
        ELBO = ELBO +  ELBO_temp;
        B_mu_all(:,:,j) = B_mu;
        B_Sigma_B_all(:,:,:,j) = B_Sigma_B;
        lambda_hat_all(:,:,j) = lambda_hat;
        B_nu_all(:,j) = B_nu;
    end

% end
