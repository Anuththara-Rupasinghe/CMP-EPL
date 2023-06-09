%% This function generates simulated spike trains given hyperparameters

% Inputs:   hyperparameters:    the true signal and noise hyperparameters
%           T_max:              the duration of each trial in seconds
%           dtbin:              the desired bin size in seconds
%           L:                  the number of trials

% Outputs:  ysamp:              the spike counts (size: time bins * trials)
%           xsamp:              the time axis in seconds (size: time bins * 1)
%           ztrue_noise:        the true noise latents (size: time bins * trials * stimuli)
%           ztrue_signal:       the true signal latents (size: time bins * stimuli)
%           ftrue:              the true firing rate in Hz (size: time bins * trials * stimuli)

function [ysamp,xsamp,ztrue_noise,ztrue_signal,ftrue] = simulate_spike_observations_multi_stimuli(hyperparameters,T_max,dtbin,L,nlfun)


    %% Generate the time axis

    K = ceil(T_max/dtbin); % number of time bins per each trial
    t_axis = (dtbin/2:dtbin:K*dtbin)*1000; % the time axis in milliseconds
    xsamp = t_axis'/1000; % time axis in seconds
    nsamp = length(xsamp); % number of time samples
    
    P = length(hyperparameters.signal.rho);% the number of different stimuli presented
    
    %% Generate the Noise Latents
    
    % Generate the noise covariance in time domain
    K_noise = get_GP_cov_from_hyp(hyperparameters,xsamp);
    
    % Generate Noise latents
    ztrue_noise = zeros(nsamp,L,P);
    
    for p = 1:P
        for l = 1:L
            ztrue_noise(:,l,p) = mvnrnd(xsamp*0,K_noise);
        end
    end

    %% Generate the Signal Latents
    
    % Specify the fourier-domain signal parameters
    nSTD = 5; % number of length scales to extend circular interval
    fdprs_signal.condthresh = 1e8; % threshold for cutting-off small frequency domain components

    ztrue_signal = zeros(nsamp,P);

    signal_len_all = hyperparameters.signal.len;
    signal_rho_all = hyperparameters.signal.rho;
        
    for p = 1:P
        
        fdprs_signal.circinterval = [0,xsamp(end)+nSTD*signal_len_all(p)]'; % circular interval 
        fdprs_signal.minlen = signal_len_all(p)*.8; % minimum length scale to consider (set higher for increased speed)

        % Extract the fourier-domain signal components
        [ddiag,Bsignal] = Krbf_fourier(xsamp,signal_len_all(p),signal_rho_all(p),fdprs_signal);
        nfreq_signal = length(ddiag);  % number of Fourier modes needed

        % Generate the Signal Latent
        fwts_signal = sqrt(ddiag).*randn(nfreq_signal,1); % function weights
        ztrue_signal(:,p) = Bsignal*squeeze(fwts_signal); % GP function values
        
    end
    %% Combine Signal and noise, pass through non-linearity and generate spikes
    
    % Combine the signal and noise components
    ztrue = 0*ztrue_noise;
    for p = 1:P
        for l = 1:L
            ztrue(:,l,p) = ztrue_signal(:,p) + ztrue_noise(:,l,p);
        end
    end
    % Pass through the non-linearity
    ftrue = nlfun(ztrue); % firing rate values
    % sample Poisson process
    ysamp = poissrnd(ftrue*dtbin); 

end
