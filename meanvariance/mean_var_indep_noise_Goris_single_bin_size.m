%% This function returns the mean, variance and the Fano Factor (FF) of spiking observations based on the GP model, across different bin sizes

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

function [mean_Goris,variance_Goris,FF_Goris] = mean_var_indep_noise_Goris_single_bin_size(r,s_vec,dtbin,bin_size)



%% Initialize the variables


%% Compute the statistics at each bin size

    % Find the downsampling factor
    downsampling_fac = round(bin_size/dtbin);
    % The number of time bins if using a sliding window approach
    K =  length(s_vec) - downsampling_fac + 1;
    % Store the mean variance estimates at each bin
    mean_temp = zeros(K,1);
    variance_temp = zeros(K,1);
    for k = 1:K
        k_range = k:k+downsampling_fac-1;
        mean_temp(k) = sum(r.*s_vec(k_range));
        variance_temp(k) =sum(r.*s_vec(k_range).*(1+ s_vec(k_range)));
    end
    
    mean_Goris = (mean_temp); % average mean across all bins
    variance_Goris = (variance_temp); % average variance across all bins
    FF_Goris = (variance_temp./mean_temp); % average Fano Factor across all bins

