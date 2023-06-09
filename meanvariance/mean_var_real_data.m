%% This function returns the empirical mean, variance and the Fano Factor (FF) of spiking observations, across different bin sizes

% Inputs:   ysamp:              the spike counts (size: time bins * trials)
%           bin_size_range:     the bin sizes that the statistics are desired to be computed (size: number of bin sizes * 1)

% Outputs:  mean_real:          the average mean of the spike count at different bin sizes (size: number of bin sizes * 1)
%           variance_real:      the average variance of the spike count at different bin sizes (size: number of bin sizes * 1)
%           FF_real:            the average Fano Factor of the spike count at different bin sizes (size: number of bin sizes * 1)
%           FF_poisson:         the average Fano Factor of the Poisson model at different bin sizes, which should be 1 (size: number of bin sizes * 1)

function [mean_real,variance_real,FF_real,FF_poisson] = mean_var_real_data(ysamp, bin_size_range, dtbin)

%% Initialize the variables

mean_real = zeros(length(bin_size_range),1);
variance_real = zeros(length(bin_size_range),1);
FF_real = zeros(length(bin_size_range),1);
FF_poisson = zeros(length(bin_size_range),1);

%% Compute the statistics at each bin size

for j = 1:length(bin_size_range)
    
    % Find the downsampling factor
    downsampling_fac = round(bin_size_range(j)/dtbin);
    % The number of time bins if using a sliding window approach
    K =  size(ysamp,1) - downsampling_fac + 1;
    % Store the downsampled spikes at the new binning
    y_downsampled = zeros(K,size(ysamp,2));
    for k = 1:K
        k_range = k:k+downsampling_fac-1;
        y_downsampled(k,:) = sum(ysamp(k_range,:),1);
    end
    mean_temp = squeeze(mean(y_downsampled,2)); % mean spike count at each time bin
    variance_temp = var(y_downsampled,0,2);  % variance spike count at each time bin
    mean_real(j) = mean(mean_temp); % average mean
    variance_real(j) = mean(variance_temp); % average variance
    FF_real(j) = mean(variance_temp./mean_temp); % average Fano Factor
    FF_poisson(j) = mean(mean_temp./mean_temp); % average Fano Facto of Poisson (assume mean = variance)

end