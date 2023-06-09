%% This function returns the empirical mean, variance and the Fano Factor (FF) of spiking observations, across different bin sizes

% Inputs:   ysamp:              the spike counts (size: time bins * trials)
%           bin_size_range:     the bin sizes that the statistics are desired to be computed (size: number of bin sizes * 1)

% Outputs:  mean_real:          the average mean of the spike count at different bin sizes (size: number of bin sizes * 1)
%           variance_real:      the average variance of the spike count at different bin sizes (size: number of bin sizes * 1)
%           FF_real:            the average Fano Factor of the spike count at different bin sizes (size: number of bin sizes * 1)
%           FF_poisson:         the average Fano Factor of the Poisson model at different bin sizes, which should be 1 (size: number of bin sizes * 1)

function [mean_real,variance_real,FF_real,FF_poisson] = mean_var_real_data_single_bin_size(ysamp, bin_size, dtbin)

%% Compute the statistics at each bin size
    
    % Find the downsampling factor
    downsampling_fac = round(bin_size/dtbin);
    % The number of time bins if using a sliding window approach
    K =  size(ysamp,1) - downsampling_fac + 1;
    % Store the downsampled spikes at the new binning
    y_downsampled = zeros(K,size(ysamp,2));
    for k = 1:K
        k_range = k:k+downsampling_fac-1;
        y_downsampled(k,:) = sum(ysamp(k_range,:),1);
    end
    mean_real = squeeze(mean(y_downsampled,2)); % mean spike count at each time bin
    variance_real = var(y_downsampled,0,2);  % variance spike count at each time bin
    FF_real = (variance_real./mean_real); % Fano Factor
    FF_poisson = (mean_real./mean_real); % average Fano Facto of Poisson (assume mean = variance)
