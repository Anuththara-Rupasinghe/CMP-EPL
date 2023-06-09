%% This script is a demo which shows the steps of inferring the latent Signal and Noise Gaussian Processes from multi-trial spiking observations by combining all stimulus presentations using the CMP-EPL inference method (data from Graft 2011)

close all;
clc;

%% Step 1: Load spike timing data and get discretize spike counts

load('sample_spike_times_all_ori'); % Sample spiking observations from Graft 2011

T_max = 2.560; % total duration of each trial in s
dtbin = 0.02; % specify the desired bin size for discretization in s
No_Orient = size(spike_times_all_ori,2); % the number of stimuli available

% Exponential nonlinearity
nlfun = @myexp; % exponential nonlinearity

% Derive the discretized spike counts from the spike times
[ysamp_temp,xsamp] = discretize_spike_train(spike_times_all_ori{1},T_max,dtbin);
ysamp_all = zeros(size(ysamp_temp,1),size(ysamp_temp,2),No_Orient);
ysamp_all(:,:,1) = ysamp_temp;
for orient = 2:No_Orient
    spike_times = spike_times_all_ori{orient};
    [ysamp_temp,~] = discretize_spike_train(spike_times,T_max,dtbin);
    ysamp_all(:,:,orient) = ysamp_temp;
end

J = 5; % Specify the number of stimuli considered (max is No_Orient)

% Picking the stimuli that had the highest, lowest responses or at random
stimuli_choice = 'highest'; % Set to random, highest or lowest
switch stimuli_choice
    case 'random'
        selected_orient = randperm(No_Orient,J);
    case 'highest'
        [~,order] = sort(squeeze(squeeze(sum(sum(ysamp_all,1),2))),'descend');
        selected_orient = order(1:J);
    case 'lowest'
        [~,order] = sort(squeeze(squeeze(sum(sum(ysamp_all,1),2))),'ascend');
        selected_orient = order(1:J);
end

ysamp = ysamp_all(:,:,selected_orient);

% Visualize the descretized spike counts
figure(1)
for l = 1:5
    for j = 1:J
        subplot(5,J,J*(l-1)+j)
        stem(xsamp,ysamp(:,l,j),'k');
        ylabel(['Trial ' num2str(l)])
        if l == 1
            title(['Spike counts stimulus ' num2str(j)])
        end
        if l == 5
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
fprintf('noise len:  %7.2f \n', estimated_hyperparameters.noise.len);
fprintf('noise rho:  %7.2f \n', estimated_hyperparameters.noise.rho);
fprintf('noise q:  %7.2f \n', estimated_hyperparameters.noise.q);
for j = 1:J
    fprintf('Stimulus %d signal len:  %7.2f \n', j, estimated_hyperparameters.signal.len(j));
    fprintf('Stimulus %d signal rho:  %7.2f \n', j, estimated_hyperparameters.signal.rho(j));
end

%% Step 3: Derive the signal and noise latents given the optimal hyper parameters, and plot sample estimates

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
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
L_plot = 1;
L_start = 10;
for j = 1:J
    for l = 1:L_plot
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 0*J*L_plot)
        stem(xsamp,ysamp(:,l+L_start,j),'k')
        title(['Stimulus ' num2str(j)])
        if j == 1
            ylabel('Spike counts','FontSize', 10)
        end
        axis([min(xsamp), max(xsamp),min(min(min(ysamp))),max(max(max(ysamp)))])
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 1*J*L_plot)
        plot(xsamp,Firing_rate_per_bin(:,l+L_start,j)/dtbin,'k', 'linewidth', linewidth)
        hold off;    
        if j == 1
            ylabel('Firing rate (Hz)','FontSize', 10)
        end
        axis([min(xsamp), max(xsamp),0,max(Firing_rate_per_bin(:,l+L_start,j)/dtbin)])        
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 2*J*L_plot)
        plot(xsamp,exp(Noise_latents(:,l+L_start,j)), 'r', 'linewidth', linewidth)
        hold off;
        if j == 1
            ylabel('Latent noise activity','FontSize', 10)
        end
        axis([min(xsamp), max(xsamp),0,max(exp(Noise_latents(:,l+L_start,j)))])        
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 3*J*L_plot)
        plot(xsamp,exp(Signal_latents(:,j)), 'b', 'linewidth', linewidth)
        hold off;
        if j == 1
            ylabel('Latent signal activity','FontSize', 10)
        end
        axis([min(xsamp), max(xsamp),0,max(exp(Signal_latents(:,j)))])
    end
end
