%% This script is a demo which shows the steps of inferring the latent Signal and Noise Gaussian Processes from multi-trial spiking observations by combining data across several stimulus presentations using the CMP-EPL inference method (simulated spiking data)

close all;
clc;

%% Step 1: Load spike timing data and get discretize spike counts

T_max = 25.6; % total duration of each trial in s
dtbin = 0.2; % specify the desired bin size for discretization
L = 50; % number of trials per stimuli
J = 2; % number of different stimuli

% Specify the true Signal Hyper-parameters
true_hyperparameters.signal.rho = 1 + (3-1).*rand(J,1);
true_hyperparameters.signal.len = 2 + (4-2).*rand(J,1);

% Specify the true Noise Hyper-parameters
true_hyperparameters.noise.rho = .8;
true_hyperparameters.noise.len = 0.3;
true_hyperparameters.noise.q = 1;

% Exponential nonlinearity
nlfun = @myexp; % exponential nonlinearity
    
% Derive the spike trains by simulations
[ysamp,xsamp,ztrue_noise,ztrue_signal,ftrue] = simulate_spike_observations_multi_stimuli(true_hyperparameters,T_max,dtbin,L,nlfun);

% Visualize the descretized spike counts
figure(1)
for j = 1:J
    subplot(9,J,j)
    plot(xsamp,ztrue_signal(:,j),'b', 'linewidth', 2)
    axis tight;
    title(['Stimulus ' num2str(j)])
end
for l = 1:8
    for j = 1:J
        subplot(9,J,J*(l-1)+j + J)
        stem(xsamp,ysamp(:,l,j),'k');
        ylabel(['Trial ' num2str(l)])
        if l == 1
            title(['Spike counts stimulus ' num2str(j)])
        end
        if l == 8
            xlabel('time (s)')
        end
    axis tight;
    end
end

%% Step 2: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method

cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
min_change_in_hyp = 10^(-3); % the threshold for checking convergence of hyper-parameters
tolerance_lambda_init = 10^(-2); % the threshold for checking onvergence of estimated firing rate

% Optimize for the hyper-parameters using the selected optimization method
estimated_hyperparameters = optimize_hyperparameters_multi_stimuli_CMP_EPL(ysamp,xsamp,dtbin,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda_init);

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
fprintf('noise len:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.noise.len, true_hyperparameters.noise.len);
fprintf('noise rho:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.noise.rho, true_hyperparameters.noise.rho);
fprintf('noise q:  %7.2f [true =%5.1f]\n', estimated_hyperparameters.noise.q, true_hyperparameters.noise.q);
for j = 1:J
    fprintf('Stimulus %d signal len:  %7.2f [true =%5.1f]\n', j, estimated_hyperparameters.signal.len(j), true_hyperparameters.signal.len(j));
    fprintf('Stimulus %d signal rho:  %7.2f [true =%5.1f]\n', j, estimated_hyperparameters.signal.rho(j), true_hyperparameters.signal.rho(j));
end

%% Step 3: Derive the signal and noise latents given the optimal hyper parameters, and plot sample estimates

tolerance_lambda_final = 10^(-4); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_q_final = [estimated_hyperparameters.noise.rho,estimated_hyperparameters.noise.len,estimated_hyperparameters.noise.q];

signal_rho_len_final = zeros(J,2);
signal_rho_len_final(:,1) = estimated_hyperparameters.signal.rho;
signal_rho_len_final(:,2) = estimated_hyperparameters.signal.len;

% Evaluate the final latent estimates
[ELBO_final,Noise_latents,Signal_latents,Firing_rate_per_bin] = derive_ELBO_VI_multi_stimuli_CMP_EPL(signal_rho_len_final,noise_rho_len_q_final,xsamp,ysamp,dtbin,tolerance_lambda_final,use_log);

% Plot and visualize sample estimates of signal and noise latents
figure(2)
linewidth = 1.5;
L_plot = 2;
L_start = 0;
contrast_val = 0.75;
for j = 1:J
    for l = 1:L_plot
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 0*J*L_plot)
        stem(xsamp,ysamp(:,l+L_start,j),'k')
        title(['Stimulus ' num2str(j) ': Trial ' num2str(l)])
        if j == 1 && l == 1
            ylabel('Spike counts','FontSize', 10)
        end
        axis([min(xsamp), max(xsamp),0,max(ysamp(:,l+L_start,j))])
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 1*J*L_plot)
        plot(xsamp,ftrue(:,l+L_start,j),'k' ,'linewidth', linewidth)
        hold on;
        plot(xsamp,Firing_rate_per_bin(:,l+L_start,j)/dtbin,'color',contrast_val*[1,1,1], 'linewidth', linewidth)
        hold off;    
        if j == 1 && l == 1
            ylabel('Firing rate (Hz)','FontSize', 10)
            legend('True','Estimated')
        end
        axis([min(xsamp), max(xsamp),0,max(ftrue(:,l+L_start,j))])
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 2*J*L_plot)
        plot(xsamp,exp(ztrue_noise(:,l+L_start,j)),'color',[1,0,0], 'linewidth', linewidth)
        hold on;
        plot(xsamp,exp(Noise_latents(:,l+L_start,j)), 'color',[1,contrast_val,contrast_val], 'linewidth', linewidth)
        hold off;
        if j == 1 && l == 1
            ylabel('Latent noise activity','FontSize', 10)
            legend('True','Estimated')
        end
        axis([min(xsamp), max(xsamp),0,max(exp(ztrue_noise(:,l+L_start,j)))])
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 3*J*L_plot)
        plot(xsamp,exp(ztrue_signal(:,j)),'color',[0,0,1], 'linewidth', linewidth)
        hold on;
        plot(xsamp,exp(Signal_latents(:,j)), 'color',[contrast_val,contrast_val,1], 'linewidth', linewidth)
        hold off;
        if j == 1 && l == 1
            ylabel('Latent signal activity','FontSize', 10)
            legend('True','Estimated')
        end
        axis([min(xsamp), max(xsamp),0,max(exp(ztrue_signal(:,j)))])
    end
end
