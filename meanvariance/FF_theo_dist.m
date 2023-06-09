%% This function empirically derives the distribution of mean, variance and Fano Factor given the signal latents of the GP model

% Inputs:   signal_latent:      the inferred signal latents (size: time bins * 1)
%           xsamp:              the time axis (size: time bins * 1)
%           hyperparameters:    the estimated hyperparameters
%           dtbin:              the bin size in ms
%           L:                  the number of trials
%           nlfun:              the non-linearity used
%           bin_size_range:     the bin sizes that the statistics are desired to be computed (size: number of bin sizes * 1)
%           no_of_repeats:      the size of the distribution

% Outputs:  mean_dist:          the distribution of the mean (size: number of bin sizes * no_of_repeats)
%           variance_dist:      the distribution of the variance (size: number of bin sizes * no_of_repeats)
%           FF_dist:            the distribution of the Fano Factor (size: number of bin sizes * no_of_repeats)

function [mean_dist,variance_dist,FF_dist] = FF_theo_dist(signal_latent,xsamp,hyperparameters,dtbin,L,nlfun,bin_size_range,no_of_repeats)
    
    % Generate the noise covariance in time domain
    K_noise = get_GP_cov_from_hyp(hyperparameters,xsamp);
    
    % Initialize the variables
    FF_dist = zeros(length(bin_size_range),no_of_repeats);
    mean_dist = FF_dist;
    variance_dist = FF_dist;
    
    for repeat = 1:no_of_repeats
        % Generate the noise latents
        ztrue_noise = zeros(length(xsamp),L);
        for l = 1:L
            ztrue_noise(:,l) = mvnrnd(xsamp*0,K_noise);
        end
        % Combine noise latents with the signal latent
        ztrue = 0*ztrue_noise;
        for l = 1:L
            ztrue(:,l) = signal_latent + ztrue_noise(:,l);
        end
        % Pass through the non-linearity
        ftrue = nlfun(ztrue); % firing rate values
        % Sample Poisson data
        ysamp = poissrnd(ftrue*dtbin); % sample Poisson noise
        % Empirically derive the statistics of the generated data
        [mean_real,variance_real,FF_real,~] = mean_var_real_data(ysamp, bin_size_range, dtbin);
        % Store the variables in each repeat
        FF_dist(:,repeat) = FF_real;
        mean_dist(:,repeat) = mean_real;
        variance_dist(:,repeat) = variance_real;
    end

end
