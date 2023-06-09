%% This function derives the time domain noise covariance, given the hyper-parameters

% Inputs: hyperparameters: hyper-parameters of the kernel
%         xsamp: the time axis in seconds (size: time bins * 1)

% Output: Knoise: the noise covariance matrix (size: time bins * time bins)

function Knoise = get_GP_cov_from_hyp(hyperparameters,xsamp)
    
    % Extract the noise hyperparameters
    noise_rho = hyperparameters.noise.rho; % noise rho
    noise_len = hyperparameters.noise.len; % noise length scale
    % If the noise exponential power q is not specified, derive using the RBF Kernel in FD domain
    if length(fieldnames(hyperparameters.noise)) == 2
        % Specify the fourier-domain noise parameters
        nSTD = 5; % number of length scales to extend circular interval
        fdprs_noise.condthresh = 1e8; % threshold for cutting-off small frequency domain components
        fdprs_noise.circinterval = [0,xsamp(end)+nSTD*noise_len]'; % circular interval 
        fdprs_noise.minlen = noise_len*.8; % minimum length scale to consider (set higher for increased speed)
        % Extract the fourier-domain noise components
        [cdiag,B_noise] = Krbf_fourier(xsamp,noise_len,noise_rho,fdprs_noise);
        % Derive the time domain noise covariance
        Knoise = B_noise * diag(cdiag) * (B_noise)';
    % If the noise exponential power q is specified, derive using the time domain formula    
    else
        noise_q = hyperparameters.noise.q;
        Knoise = noise_rho*exp(-0.5*abs((xsamp(:)-xsamp(:)')/noise_len).^noise_q);
    end

% end
