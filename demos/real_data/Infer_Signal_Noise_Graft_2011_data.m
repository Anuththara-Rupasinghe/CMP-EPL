%% This script is a demo which shows the steps of inferring the latent Signal and Noise Gaussian Processes from multi-trial spiking observations, for a single stimulus presentation using the CMP-EPL inference method (data from Graft 2011)

close all;
clc;

%% Step 1: Load spike timing data and get discretize spike counts

load('sample_spike_times'); % Sample spiking observations from Graft 2011

T_max = 2.560; % total duration of each trial in s
dtbin = 0.02; % specify the desired bin size for discretization in s

% Exponential nonlinearity
nlfun = @myexp; % exponential nonlinearity

% Derive the discretized spike counts from the spike times
[ysamp,xsamp] = discretize_spike_train(spike_times,T_max,dtbin);

% Visualize the descretized spike counts
figure(1)
for l = 1:size(ysamp,2)
    subplot(10,5,l)
    stem(xsamp,ysamp(:,l),'k');
    ylabel(['Trial ' num2str(l)])
    if l < 6
        title('Spike counts')
    end
    if l > 45
        xlabel('time (s)')
    end
    axis tight;
end

%% Step 2: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method

cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
min_change_in_hyp = 10^(-3); % the threshold for checking convergence of hyper-parameters
tolerance_lambda_init = 10^(-2); % the threshold for checking onvergence of estimated firing rate

% Optimize for the hyper-parameters using the selected optimization method
estimated_hyperparameters = optimize_hyperparameters_CMP_EPL(ysamp,xsamp,dtbin,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda_init);

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
fprintf('noise len:  %7.2f \n', estimated_hyperparameters.noise.len);
fprintf('noise rho:  %7.2f \n', estimated_hyperparameters.noise.rho);
fprintf('noise q:  %7.2f \n', estimated_hyperparameters.noise.q);
fprintf('signal len:  %7.2f \n', estimated_hyperparameters.signal.len);
fprintf('signal rho:  %7.2f \n', estimated_hyperparameters.signal.rho);

%% Step 3: Derive the signal and noise latents given the optimal hyper parameters, and plot sample estimates

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_q_final = [estimated_hyperparameters.noise.rho,estimated_hyperparameters.noise.len,estimated_hyperparameters.noise.q];
signal_rho_len_final = [estimated_hyperparameters.signal.rho, estimated_hyperparameters.signal.len];

% Evaluate the final latent estimates
[ELBO_final,Noise_latents,Signal_latents,Firing_rate_per_bin] = derive_ELBO_VI_CMP_EPL(signal_rho_len_final,noise_rho_len_q_final,xsamp,ysamp,dtbin,tolerance_lambda_final,use_log);

% Plot and visualize sample estimates of signal and noise latents
linewidth = 1.5;
L_plot = 10;
L_start = 30;
figure(3)
for l = 1:L_plot
    subplot(L_plot,4,4*(l-1)+1)
    stem(xsamp,ysamp(:,l+L_start),'k')
    ylabel(['trial ' num2str(l+L_start)])
    if l == 1
        title('Spike counts','FontSize', 10)
    end
    subplot(L_plot,4,4*(l-1)+2)
    plot(xsamp,Firing_rate_per_bin(:,l+L_start)/dtbin,'k' ,'linewidth', linewidth) 
    if l == 1
        title('Firing rate (Hz)','FontSize', 10)
    end
    subplot(L_plot,4,4*(l-1)+3)
    plot(xsamp,exp(Noise_latents(:,l+L_start)),'color',[1,0,0], 'linewidth', linewidth)
    if l == 1
        title('Latent noise activity','FontSize', 10)
    end
    subplot(L_plot,4,4*(l-1)+4)
    plot(xsamp,exp(Signal_latents),'color',[0,0,1], 'linewidth', linewidth)
    if l == 1
        title('Latent signal activity','FontSize', 10)
    end
end

%% Step 4: Derive the mean-variance relationship at the smallest bin size

% Specify the range of bin sizes considered
bin_size = dtbin;
% Set this logical variable to true if using the theoretical covariance,false if using the empirical covariance
theoretical = true;

% Compute the mean, variance and Fano-Factor of real data empirically
[mean_real_dtbin,variance_real_dtbin,FF_real_dtbin,FF_poisson_dtbin] = mean_var_real_data_single_bin_size(ysamp, bin_size, dtbin);
% Compute the mean, variance and Fano-Factor using the analytical expressions of the Poisson-GP model
[mean_GP_dtbin,variance_GP_dtbin,FF_GP_dtbin] = mean_var_CMP_EPL_single_bin_size(Signal_latents,Noise_latents,estimated_hyperparameters,xsamp,dtbin,bin_size,theoretical);

figure(4)
subplot(1,2,1)
plot(xsamp,mean_GP_dtbin, 'b', 'linewidth', linewidth)
hold on;
plot(xsamp,variance_GP_dtbin, 'r', 'linewidth', linewidth)
hold on;
plot(xsamp,FF_GP_dtbin, 'k', 'linewidth', linewidth)
hold on;
plot(xsamp,mean_real_dtbin, 'b--', 'linewidth', linewidth)
hold on;
plot(xsamp,variance_real_dtbin, 'r--', 'linewidth', linewidth)
hold on;
plot(xsamp,FF_real_dtbin, 'k--', 'linewidth', linewidth)
hold off;
xlabel('time (s)')
legend('mean est','variance est','Fano Factor est','mean data','variance data','Fano Factor data')
axis tight;

%% Step 5: Derive the mean-variance relationship at different bin sizes

% Specify the range of bin sizes considered
bin_size_all = (dtbin:dtbin:floor(T_max/2));
% Set this logical variable to true if using the theoretical covariance,false if using the empirical covariance
theoretical = true;

% Compute the mean, variance and Fano-Factor of real data empirically
[mean_real,variance_real,FF_real,FF_poisson] = mean_var_real_data(ysamp, bin_size_all, dtbin);
% Compute the mean, variance and Fano-Factor using the analytical expressions of the Poisson-GP model
[mean_GP,variance_GP,FF_GP] = mean_var_CMP_EPL(Signal_latents,Noise_latents,estimated_hyperparameters,xsamp,dtbin,bin_size_all,theoretical);

% Visualize using plotting
figure(4)
subplot(1,2,2)
plot(bin_size_all,FF_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,FF_poisson,'k','linewidth',1.5)
hold on;
plot(bin_size_all,FF_GP,'r','linewidth',1.5)
hold off
legend('Real data', 'Poisson','GP')
xlabel('bin size (ms)')
ylabel('Fano Factor')
% ylabel('Variance')
axis tight;
axis([min(bin_size_all), max(bin_size_all), 0, max([FF_GP;FF_real])+1])

%% Step 6: Derive the confidence bounds of the mean-variance-Fano Factor estimates

% Specify the size of the distribution
no_of_repeats = 1000;
% Specify the desired confidence bounds
conf_level = 0.95;
% Derive the distribution of the statistics
[mean_dist,variance_dist,FF_dist] = FF_theo_dist(Signal_latents,xsamp,estimated_hyperparameters,dtbin,size(ysamp,2),nlfun,bin_size_all,no_of_repeats);
% Get the confidence bounds of each statistic
[conf_min_FF,conf_max_FF,conf_bin_size_FF] = get_conf_bounds(FF_dist,bin_size_all,conf_level);
[conf_min_mean,conf_max_mean,conf_bin_size_mean] = get_conf_bounds(mean_dist,bin_size_all,conf_level);
[conf_min_var,conf_max_var,conf_bin_size_var] = get_conf_bounds(variance_dist,bin_size_all,conf_level);

figure(5)
% Confidence Bounds of the mean
subplot(2,3,1)
fill([conf_bin_size_mean';flipud(conf_bin_size_mean')],[conf_min_mean';flipud(conf_max_mean')],[1,0.9,0.9],'linestyle','none');hold on;
hold on;
plot(bin_size_all,mean_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,mean_GP,'r','linewidth',1.5)
hold off
xlabel('bin size (ms)')
ylabel('Mean')
axis tight;
title(['Empirical ' num2str(95) ' % Confidence Bounds Mean'])
% Confidence Bounds of the variance
subplot(2,3,2)
fill([conf_bin_size_var';flipud(conf_bin_size_var')],[conf_min_var';flipud(conf_max_var')],[1,0.9,0.9],'linestyle','none');hold on;
hold on;
plot(bin_size_all,variance_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,variance_GP,'r','linewidth',1.5)
hold off
xlabel('bin size (ms)')
ylabel('Variance')
axis tight;
title(['Empirical ' num2str(95) ' % Confidence Bounds Variance'])
% Confidence Bounds of the Fano Factor
subplot(2,3,3)
fill([conf_bin_size_FF';flipud(conf_bin_size_FF')],[conf_min_FF';flipud(conf_max_FF')],[1,0.9,0.9],'linestyle','none');hold on;
hold on;
plot(bin_size_all,FF_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,FF_GP,'r','linewidth',1.5)
hold off
xlabel('bin size (ms)')
ylabel('Fano Factor')
axis tight;
title(['Empirical ' num2str(95) ' % Confidence Bounds Fano Factor'])
subplot(2,3,4)
% Individual repeats of the mean
for repeats = 1:no_of_repeats
    temp = mean_dist(:,repeats);
    temp_valid = ~isnan(temp);
    plot(bin_size_all(temp_valid),temp(temp_valid),'Color',[1,0.9,0.9])
    hold on;
end
plot(bin_size_all,mean_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,mean_GP,'r','linewidth',1.5)
hold off
xlabel('bin size (ms)')
ylabel('Mean')
axis tight;
title(['Individual repeats: Mean'])
subplot(2,3,5)
% Individual repeats of the variance
for repeats = 1:no_of_repeats
    temp = variance_dist(:,repeats);
    temp_valid = ~isnan(temp);
    plot(bin_size_all(temp_valid),temp(temp_valid),'Color',[1,0.9,0.9])
    hold on;
end
plot(bin_size_all,variance_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,variance_GP,'r','linewidth',1.5)
hold off
xlabel('bin size (ms)')
ylabel('Variance')
axis tight;
title(['Individual repeats: Variance'])
subplot(2,3,6)
% Individual repeats of the Fano-Factor
for repeats = 1:no_of_repeats
    temp = FF_dist(:,repeats);
    temp_valid = ~isnan(temp);
    plot(bin_size_all(temp_valid),temp(temp_valid),'Color',[1,0.9,0.9])
    hold on;
end
plot(bin_size_all,FF_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,FF_GP,'r','linewidth',1.5)
hold off
xlabel('bin size (ms)')
ylabel('Fano Factor')
axis tight;
title(['Individual repeats: FF'])
