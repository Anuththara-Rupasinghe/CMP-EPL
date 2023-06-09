%% This function computes the ELBO given hyper-parameters for the Poisson Signal GP model (no noise component)

% Inputs:   signal_rho_len: a vector of length 2 containing [signal rho, signal length]
%           ysamp:  the spike counts (size: time bins * trials)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts
%           use_log: a logical variable indicating whether the inference is performed in log-domain

% Outputs:  ELBO: Derived Evidence Lower BOund
%           lambda_hat_all: Estimated firing rates per bin (size: time bins * trials)

function [ELBO,B_nu,lambda_hat_all] = derive_ELBO_VI_Poisson_GP(signal_rho_len,xsamp,ysamp,dtbin,use_log)  
    
    % Transform the hyper-parameters if using the log-domain
    if use_log 
        signal_rho_len = exp(signal_rho_len);
    end
    
    % Extract the hyper-parameters
    signal_rho = (signal_rho_len(1));
    signal_len = (signal_rho_len(2));
    
    % Specify the fourier-domain signal parameters
    nSTD = 5; % number of length scales to extend circular interval
    fdprs_signal.condthresh = 1e8; % threshold for cutting-off small frequency domain components
    fdprs_signal.circinterval = [0,xsamp(end)+nSTD*signal_len]'; % circular interval 
    fdprs_signal.minlen = signal_len*.8; % minimum length scale to consider (set higher for increased speed)
    
    % Extract the fourier-domain signal components
    [ddiag,B_signal] = Krbf_fourier(xsamp,signal_len,signal_rho,fdprs_signal);
    ddiag_inv = (1./ddiag);
    ddiag_inv_mat = diag(ddiag_inv);
    
    % Initialize variables
    M = size(B_signal,2); % Dimensionality of Signal Latents in Frequency domain
    L = size(ysamp,2); % Number of trials    
    nu = zeros(M,1); % Estimated mean Signal Latents in Frequency domain
    B_nu = B_signal*nu; % Estimated mean Signal Latents in Time domain
    Omegha = zeros(M,M); % Estimated Signal Covariance in Variational Inference
    lambda_hat_all = ysamp;
    for l = 1:L
        lambda_hat_all(:,l) = 0.1*mean(ysamp,2);
    end
    iter_Newton_max = 10^5; % Maximum number of Newton's iterations permitted    
            
    % Estimate Signal Latents using
    % Newton-Rhapson method
    error = 1;
    iter_Newton = 1;        
    while (error > 10^(-5)) && (iter_Newton < iter_Newton_max)
        %The gradient of the objective function
        gradient = B_signal'*(sum(ysamp,2) - sum(lambda_hat_all,2)) - ddiag_inv_mat*nu;
        %The Hessian matrix of the objective function
        Hessian = -B_signal'*diag(sum(lambda_hat_all,2))*B_signal - ddiag_inv_mat;
        % Newton's update
        Omegha = - Hessian \ eye(M);         
        nu = nu + Omegha*gradient;
        % Update the Signal latent in time-domain estimate
        B_nu = B_signal*nu;
        error = sum(abs(gradient));
        % Update the firing rate estimate
        for l = 1:L
            lambda_hat_all(:,l) = dtbin*exp(B_nu + diag(B_signal*Omegha*B_signal')/2);
        end
        iter_Newton = iter_Newton + 1;
    end      
    
    % Derive the ELBO
    % The Signal Component of the ELBO
    ELBO = -0.5*sum(log(abs(ddiag))) -0.5*ddiag_inv'*(nu.^2 + diag(Omegha)) + 0.5*logdet(Omegha);
    for l = 1:L
    ELBO = ELBO - sum(lambda_hat_all(:,l)) + (squeeze(ysamp(:,l)))'*(B_nu); 
    end
    % Use the negative of ELBO as we are minimizing the cost function (= maximizing the ELBO)
    ELBO = -ELBO;
   
% end
