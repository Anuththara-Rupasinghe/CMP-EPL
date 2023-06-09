%% This function computes the ELBO given hyper-parameters for the RBF kernel noise covariance model (CMP - RBF) for a single stimulus

% Inputs:   signal_rho_len: a vector of length 2 containing [signal rho, signal length]
%           noise_rho_len: a vector of length 2 containing [noise rho, noise length]
%           ysamp:  the spike counts (size: time bins * trials)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts
%           tolerance_lambda: the threshold for checking convergence of estimated firing rate
%           use_log: a logical variable indicating whether the inference is performed in log-domain

% Outputs:  ELBO: Derived Evidence Lower BOund
%           B_mu_all: Estimated noise latents (size: time bins * trials)
%           B_nu: Estimated signal latents (size: time bins * 1)
%           lambda_hat_all: Estimated firing rates per bin (size: time bins * trials)
%           B_Sigma_B_all: Estimate noise covariance in time domain (size: time bins * time bins * trials)

function [ELBO,B_mu_all,B_nu,lambda_hat_all,B_Sigma_B_all] = derive_ELBO_VI_CMP_RBF(signal_rho_len,noise_rho_len,xsamp,ysamp,dtbin,tolerance_lambda,use_log)  

    % Transform the hyper-parameters if using the log-domain
    if use_log 
        noise_rho_len = exp(noise_rho_len);
        signal_rho_len = exp(signal_rho_len);
    end

    % Extract the hyper-parameters
    noise_rho = (noise_rho_len(1));
    noise_len = (noise_rho_len(2));
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

    % Specify the fourier-domain noise parameters
    nSTD = 5; % number of length scales to extend circular interval
    fdprs_noise.condthresh = 1e8; % threshold for cutting-off small frequency domain components
    fdprs_noise.circinterval = [0,xsamp(end)+nSTD*noise_len]'; % circular interval 
    fdprs_noise.minlen = noise_len*.8; % minimum length scale to consider (set higher for increased speed)

    % Extract the fourier-domain noise components
    [cdiag,B_noise] = Krbf_fourier(xsamp,noise_len,noise_rho,fdprs_noise);
    cdiag_inv = (1./cdiag);
    cdiag_inv_mat = diag(cdiag_inv);
    
    % Initialize variables
    N = size(B_noise,2); % Dimensionality of Noise Latents in Frequency domain
    M = size(B_signal,2); % Dimensionality of Signal Latents in Frequency domain
    L = size(ysamp,2); % Number of trials 
    K = size(ysamp,1); % Number of time bins
    nu = zeros(M,1); % Estimated mean Signal Latents in Frequency domain
    B_nu = B_signal*nu; % Estimated mean Signal Latents in Time domain
    B_mu_all = zeros(size(B_noise,1),L); % Estimated mean Noise Latents in Time domain
    mu_all = zeros(N,L);% Estimated mean Noise Latents in Frequency domain
    Omegha = zeros(M,M); % Estimated Signal Covariance in Variational Inference
    Sigma = zeros(N,N);
    Sigma_all = zeros(N,N,L); % Estimated Noise Covariance in Variational Inference
    B_Sigma_B_all =  zeros(K,K,L); % Estimated Noise Covariance in Variational Inference in time domain
    lambda_hat_all = mean(mean(ysamp))*ones(size(B_noise,1),L); % Estimated firing rate per bin
    iter_Newton_max = 10^5; % Maximum number of Newton's iterations permitted    
    iter_cordinate_desc_max = 10^5; % Maximum number of cordinate descend iterations permitted
    
    % Initialize the variable checked for convergence
    iter_cordinate_desc = 1;
    error_lambda = 1;
    
    % Update the signal and noise latent components using cordinate descend
    while (iter_cordinate_desc < iter_cordinate_desc_max)&&(error_lambda > tolerance_lambda)
        % Initialize the variable checked for convergence
        lambda_hat_all_prev = lambda_hat_all;
        
        % Estimate Signal Latents conditioned on the noise latents using
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
            error = max(abs(gradient));
%             error = sum(abs(gradient));
            % Update the firing rate estimate
            for l = 1:L
                lambda_hat_all(:,l) = dtbin*exp(B_mu_all(:,l) + diag(B_noise*Sigma_all(:,:,l)*B_noise')/2 + B_nu + diag(B_signal*Omegha*B_signal')/2);
            end
            iter_Newton = iter_Newton + 1;
        end
        
        % Estimate Noise Latents conditioned on the signal latents using
        % Newton-Rhapson method (can be done for each trial independantly)
        for l = 1:L 
            mu = mu_all(:,l);
            lambda_hat = lambda_hat_all(:,l);
            error = 1;
            iter_Newton = 1;
            while (error > 10^(-2))  &&(iter_Newton < iter_Newton_max)
                %The gradient of the objective function
                gradient = B_noise'*(squeeze(ysamp(:,l)) - lambda_hat) - cdiag_inv_mat*mu;
                %The Hessian matrix of the objective function
                Hessian = -B_noise'*diag(lambda_hat)*B_noise - cdiag_inv_mat;
                % Newton's update
                Sigma = - Hessian \ eye(N);
                mu = mu + Sigma*gradient;
                B_mu = B_noise*mu;
                lambda_hat = dtbin*exp(B_mu +  B_nu + diag(B_noise*Sigma*B_noise')/2 + diag(B_signal*Omegha*B_signal')/2);
                error = max(abs(gradient));
                iter_Newton = iter_Newton + 1;
            end
            % Update the firing rate estimate
            lambda_hat_all(:,l) = lambda_hat;
            % Update the noise latent in time-domain estimate
            B_mu_all(:,l) = B_mu;
            % Update the noise covariance in frequency-domain estimate
            Sigma_all(:,:,l) = Sigma;
            % Update the noise covariance in time-domain estimate
            B_Sigma_B_all(:,:,l) = B_noise*Sigma*B_noise';
            % Update the noise latent in frequency-domain estimate            
            mu_all(:,l) = mu;
        end
        % evaluate the convergence criteria
        iter_cordinate_desc = iter_cordinate_desc + 1;
        error_lambda = sum(sum(abs(lambda_hat_all_prev - lambda_hat_all)))/sum(sum(abs(lambda_hat_all)));
    end
    
    % Derive the ELBO
    % The Signal Component of the ELBO
    ELBO = -0.5*sum(log(abs(ddiag))) -0.5*ddiag_inv'*(nu.^2 + diag(Omegha)) + 0.5*logdet(Omegha);
    % The Noise Component of the ELBO
    ELBO = ELBO - 0.5*L*sum(log(abs(cdiag)));  
    for l = 1:L
    ELBO = ELBO - sum(lambda_hat_all(:,l)) + (squeeze(ysamp(:,l)))'*(B_mu_all(:,l)+B_nu) -0.5*cdiag_inv'*(mu_all(:,l).^2 + diag(squeeze(Sigma_all(:,:,l)))) + 0.5*logdet(squeeze(Sigma_all(:,:,l))); 
    end
    % Use the negative of ELBO as we are minimizing the cost function (= maximizing the ELBO)
    ELBO = -ELBO;
   
% end
