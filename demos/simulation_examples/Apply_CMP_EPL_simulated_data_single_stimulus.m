%% This script is a demo which shows the steps of inferring the latent Signal and Noise Gaussian Processes from multi-trial spiking observations for a single stimulus presentation using the CMP-EPL inference method (simulated spiking data)

close all;
clc;

%% Step 1: Load spike timing data and get discretize spike counts

T_max = 25.6; % total duration of each trial in s
dtbin = 0.2; % specify the desired bin size for discretization
L = 50; % specify the number of trials available for inference

% Specify the true Signal Hyper-parameters
true_hyperparameters.signal.rho = 2;
true_hyperparameters.signal.len = 2;

% Specify the true Noise Hyper-parameters
true_hyperparameters.noise.rho = 0.8;
true_hyperparameters.noise.len = 0.3;
true_hyperparameters.noise.q = 1;

% Exponential nonlinearity
nlfun = @myexp; % exponential nonlinearity
    
% Derive the spike trains by simulations
[ysamp,xsamp,ztrue_noise,ztrue_signal,ftrue] = simulate_spike_observations(true_hyperparameters,T_max,dtbin,L,nlfun);

% Visualize the descretized spike counts
figure(1)
for l = 1:size(ysamp,2)
    subplot(ceil(L/5),5,l)
    stem(xsamp,ysamp(:,l),'k');
    ylabel(['Trial ' num2str(l)])
    if l < 6
        title('Spike counts')
    end
    if l > (ceil(L/5)-1)*5
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
fprintf('noise len:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.noise.len, true_hyperparameters.noise.len);
fprintf('noise rho:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.noise.rho, true_hyperparameters.noise.rho);
fprintf('noise q:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.noise.q, true_hyperparameters.noise.q);
fprintf('signal len:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.signal.len, true_hyperparameters.signal.len);
fprintf('signal rho:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.signal.rho, true_hyperparameters.signal.rho);

%% Step 3: Derive the signal and noise latents given the optimal hyper parameters, and plot sample estimates

tolerance_lambda_final = 10^(-3); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_q_final = [estimated_hyperparameters.noise.rho,estimated_hyperparameters.noise.len,estimated_hyperparameters.noise.q];
signal_rho_len_final = [estimated_hyperparameters.signal.rho, estimated_hyperparameters.signal.len];

% Evaluate the final latent estimates
[ELBO_final,Noise_latents,Signal_latents,Firing_rate_per_bin] = derive_ELBO_VI_CMP_EPL(signal_rho_len_final,noise_rho_len_q_final,xsamp,ysamp,dtbin,tolerance_lambda_final,use_log);

% Plot and visualize sample estimates of signal and noise latents
linewidth = 1.5;
L_plot = 10;
L_start = 10;
contrast_val = 0.75;
figure(2)
for l = 1:L_plot
    subplot(L_plot,4,4*(l-1)+1)
    stem(xsamp,ysamp(:,l+L_start),'k')
    ylabel(['trial ' num2str(l+L_start)])
    if l == 1
        title('Spike counts','FontSize', 10)
    end
    subplot(L_plot,4,4*(l-1)+2)
    plot(xsamp,ftrue(:,l+L_start),'k' ,'linewidth', linewidth)
    hold on;
    plot(xsamp,Firing_rate_per_bin(:,l+L_start)/dtbin, 'color', contrast_val*[1,1,1], 'linewidth', linewidth)
    hold off;    
    if l == 1
        title('Firing rate (Hz)','FontSize', 10)
        legend('True','Estimated')
    end
    subplot(L_plot,4,4*(l-1)+3)
    plot(xsamp,exp(ztrue_noise(:,l+L_start)),'color',[1,0,0], 'linewidth', linewidth)
    hold on;
    plot(xsamp,exp(Noise_latents(:,l+L_start)), 'color', [1,contrast_val,contrast_val], 'linewidth', linewidth)
    hold off;
    if l == 1
        title('Latent noise activity','FontSize', 10)
        legend('True','Estimated')
    end
    subplot(L_plot,4,4*(l-1)+4)
    plot(xsamp,exp(ztrue_signal),'color',[0,0,1], 'linewidth', linewidth)
    hold on;
    plot(xsamp,exp(Signal_latents), 'color',[contrast_val,contrast_val,1], 'linewidth', linewidth)
    hold off;
    if l == 1
        title('Latent signal activity','FontSize', 10)
        legend('True','Estimated')
    end
end

%% Step 4: Derive the mean-variance relationship at different bin sizes

% Specify the range of bin sizes considered
bin_size_all = (dtbin:dtbin:floor(T_max/2));
% Set this logical variable to true if using the theoretical covariance,false if using the empirical covariance
theoretical = true;

% Compute the mean, variance and Fano-Factor of real data empirically
[mean_real,variance_real,FF_real,FF_poisson] = mean_var_real_data(ysamp, bin_size_all, dtbin);

% Compute the mean, variance and Fano-Factor using the analytical expressions of the Poisson-GP model and estimated parameters
[mean_GP,variance_GP,FF_GP] = mean_var_CMP_EPL(Signal_latents,Noise_latents,estimated_hyperparameters,xsamp,dtbin,bin_size_all,theoretical);

% Compute the mean, variance and Fano-Factor using the analytical expressions of the Poisson-GP model and true parameters
[mean_GP_true,variance_GP_true,FF_GP_true] = mean_var_CMP_EPL(ztrue_signal,ztrue_noise,true_hyperparameters,xsamp,dtbin,bin_size_all,theoretical);

% Visualize using plotting
figure(3)
plot(bin_size_all,FF_real,'b','linewidth',1.5)
hold on;
plot(bin_size_all,FF_GP_true,'k','linewidth',1.5)
hold on;
plot(bin_size_all,FF_GP,'r','linewidth',1.5)
hold off
legend('Real data', 'CMP-EPL true parameters','CMP-EPL estimated parameters')
xlabel('bin size (ms)')
ylabel('Fano Factor')
% ylabel('Variance')
axis tight;
axis([min(bin_size_all), max(bin_size_all), 0, max([FF_GP;FF_real;FF_GP_true])+1])