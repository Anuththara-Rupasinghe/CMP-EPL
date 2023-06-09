%% This function computes the ELBO given hyper-parameters of the CMP Matern Model for the case of single stimulus

% Inputs:   signal_rho_len: a vector of length 2 containing [signal rho, signal length]
%           noise_rho_len_q: a vector of length 3 containing [noise rho, noise length, noise q]
%           ysamp:  the spike counts (size: time bins * trials)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts
%           tolerance_lambda: the threshold for checking convergence of estimated firing rate
%           use_log: a logical variable indicating whether the inference is performed in log-domain
%           Matern_order: specify the order of the Matern kernel (0 or 1)

% Outputs:  ELBO: Derived Evidence Lower BOund
%           mu_all: Estimated noise latents (size: time bins * trials)
%           B_nu: Estimated signal latents (size: time bins * 1)
%           lambda_hat_all: Estimated firing rates per bin (size: time bins * trials)
%           Sigma_all: Estimated noise covariances of the variational distribution (size: time bins * time bins * trials)


function [ELBO,mu_all,B_nu,lambda_hat_all,Sigma_all] = derive_ELBO_VI_CMP_Matern(signal_rho_len,noise_rho_len,xsamp,ysamp,dtbin,tolerance_lambda,use_log,Matern_order)  
    
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

    % Extract the time-domain noise covariance
    K_matrix_noise = get_Matern_cov(xsamp,noise_len,noise_rho,Matern_order);
    K_matrix_noise_inv = pinv(K_matrix_noise);

    % Initialize variables
    M = size(B_signal,2); % Dimensionality of Signal Latents in Frequency domain
    L = size(ysamp,2); % Number of trials 
    K = size(ysamp,1); % Number of time bins
    nu = zeros(M,1); % Estimated mean Signal Latents in Frequency domain
    B_nu = B_signal*nu; % Estimated mean Signal Latents in Time domain
    sum_of_latents_init = max(log(ysamp/dtbin),-20);
    mu_all = zeros(K,L); % Estimated mean Noise Latents in Time domain
    for l = 1:L
        mu_all(:,l) = sum_of_latents_init(:,l) -  B_nu;
    end
    Omegha = zeros(M,M); % Estimated Signal Covariance in Variational Inference
    Sigma = zeros(K,K);
    Sigma_all = zeros(K,K,L); % Estimated Noise Covariance in Variational Inference
    lambda_hat_all = ysamp; % Initialize the Firing rate with the spiking observations
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
            % Update the firing rate estimate
            for l = 1:L
                lambda_hat_all(:,l) = dtbin*exp(mu_all(:,l) + diag(Sigma_all(:,:,l))/2 + B_nu + diag(B_signal*Omegha*B_signal')/2);
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
                gradient = (squeeze(ysamp(:,l)) - lambda_hat) - K_matrix_noise_inv*mu;
                %The Hessian matrix of the objective function
                Hessian = -diag(lambda_hat) - K_matrix_noise_inv;
                % Newton's update
                Sigma = - Hessian \ eye(K);
                mu = mu + Sigma*gradient;
                % Update the firing rate estimate
                lambda_hat = dtbin*exp(mu +  B_nu + diag(Sigma)/2 + diag(B_signal*Omegha*B_signal')/2);
                error = max(abs(gradient));
                iter_Newton = iter_Newton + 1;
            end
            % Update the firing rate estimate
            lambda_hat_all(:,l) = lambda_hat;
            % Update the noise latent in time-domain estimate
            mu_all(:,l) = mu;
            % Update the noise covariance in time-domain estimate
            Sigma_all(:,:,l) = Sigma;
        end
        % evaluate the convergence criteria
        iter_cordinate_desc = iter_cordinate_desc + 1;
        error_lambda = sum(sum(abs(lambda_hat_all_prev - lambda_hat_all)))/sum(sum(abs(lambda_hat_all)));
    end
    
    % Derive the ELBO
    % The Signal Component of the ELBO
    ELBO = -0.5*sum(log(abs(ddiag))) -0.5*ddiag_inv'*(nu.^2 + diag(Omegha)) + 0.5*logdet(Omegha);
    % The Noise Component of the ELBO
    ELBO = ELBO - 0.5*L*((logdet(K_matrix_noise)));  
    for l = 1:L
    ELBO = ELBO - sum(lambda_hat_all(:,l)) + (squeeze(ysamp(:,l)))'*(mu_all(:,l)+B_nu) -0.5*trace(K_matrix_noise_inv'*(squeeze(mu_all(:,l))*squeeze(mu_all(:,l))' + squeeze(Sigma_all(:,:,l)))) + 0.5*logdet(squeeze(Sigma_all(:,:,l))); 
    end
    % Use the negative of ELBO as we are minimizing the cost function (= maximizing the negative ELBO)
    ELBO = -ELBO;
   
% end
