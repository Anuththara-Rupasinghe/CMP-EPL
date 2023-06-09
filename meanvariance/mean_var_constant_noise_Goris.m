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

function [mean_Goris_cons,variance_Goris_cons,FF_Goris_cons] = mean_var_constant_noise_Goris(lambda_x,var_G,dtbin,bin_size_range)


%% Derive the noise covariance matrix theoretically or empirically


%% Initialize the variables

mean_Goris_cons = zeros(length(bin_size_range),1);
variance_Goris_cons = zeros(length(bin_size_range),1);
FF_Goris_cons = zeros(length(bin_size_range),1);


%% Compute the statistics at each bin size

for j = 1:length(bin_size_range)
    % Find the downsampling factor
    downsampling_fac = round(bin_size_range(j)/dtbin);
    % The number of time bins if using a sliding window approach
    K =  size(lambda_x,1) - downsampling_fac + 1;
    % Store the mean variance estimates at each bin
    mean_temp = zeros(K,1);
    variance_temp = zeros(K,1);
    for k = 1:K
        k_range = k:k+downsampling_fac-1;
        mean_temp(k) = sum(lambda_x(k_range));
        variance_temp(k) = mean_temp(k);
        for i = 1:length(k_range)
            for p = 1:length(k_range)
                t_i = k_range(i); t_p = k_range(p);
                variance_temp(k) = variance_temp(k) + lambda_x(t_i).*lambda_x(t_p).*(var_G + 1);
            end
        end
        variance_temp(k) = variance_temp(k) -(mean_temp(k)).^2;
    end
    
    mean_Goris_cons(j) = mean(mean_temp); % average mean across all bins
    variance_Goris_cons(j) = mean(variance_temp); % average variance across all bins
    FF_Goris_cons(j) = mean(variance_temp./mean_temp); % average Fano Factor across all bins

end