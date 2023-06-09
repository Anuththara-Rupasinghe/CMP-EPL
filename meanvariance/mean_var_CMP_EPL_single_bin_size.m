%% This function returns the mean, variance and the Fano Factor (FF) of spiking observations based on the GP model, at a single bin size

% Inputs:   signal_latents:     the inferred signal latents (size: time bins * 1)
%           noise_latents:      the inferred noise latents (size: time bins * trial)
%           hyperparameters:    the estimated hyperparameters
%           bin_size_range:     the bin sizes that the statistics are desired to be computed (size: number of bin sizes * 1)
%           xsamp:              the time axis (size: time bins * 1)
%           dtbin:              the bin size in ms
%           theoretical:        a logical operator set to true if computing the theoretical covariance, false if computing the empirical covariace

% Outputs:  mean_GP:          the average mean of the spike count at different bin sizes (size: number of bin sizes * 1)
%           variance_GP:      the average variance of the spike count at different bin sizes (size: number of bin sizes * 1)
%           FF_GP:            the average Fano Factor of the spike count at different bin sizes (size: number of bin sizes * 1)

function [mean_GP,variance_GP,FF_GP] = mean_var_CMP_EPL_single_bin_size(signal_latents,noise_latents,hyperparameters,xsamp,dtbin,bin_size,theoretical)


%% Derive the noise covariance matrix theoretically or empirically

    if theoretical
        Knoise = get_GP_cov_from_hyp(hyperparameters,xsamp);
    else 
        Knoise = cov(noise_latents');
    end
    Knoise_diag = diag(Knoise);

%% Compute the statistics at each bin size


    % Find the downsampling factor
    downsampling_fac = round(bin_size/dtbin);
    % The number of time bins if using a sliding window approach
    K =  size(xsamp,1) - downsampling_fac + 1;
    % Store the mean variance estimates at each bin
    mean_GP = zeros(K,1);
    variance_GP = zeros(K,1);
    for k = 1:K
        k_range = k:k+downsampling_fac-1;
        mean_GP(k) = sum(exp(signal_latents(k_range)).*exp(Knoise_diag(k_range)/2).*dtbin);
        variance_GP(k) = mean_GP(k);
        for i = 1:length(k_range)
            for p = 1:length(k_range)
                t_i = k_range(i); t_p = k_range(p);
                variance_GP(k) = variance_GP(k) + exp(signal_latents(t_i)+signal_latents(t_p)).*exp((Knoise(t_i,t_i)+Knoise(t_p,t_p) + 2*Knoise(t_i,t_p))/2).*dtbin^2;
            end
        end
        variance_GP(k) = variance_GP(k) -(mean_GP(k)).^2;
    end
    
    FF_GP = variance_GP./mean_GP; % average Fano Factor across all bins
