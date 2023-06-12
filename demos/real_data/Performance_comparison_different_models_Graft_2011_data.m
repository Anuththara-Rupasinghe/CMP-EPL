%% This script is a demo which compares the performance of different models and methods for Signal and Noise inference (data from Graft 2011)

clear all;
close all;
clc;

%% Step 1: Load spike timing data and get discretize spike counts

load('sample_spike_times_all_ori'); % Sample spiking observations from Graft 2011

T_max = 2.560; % total duration of each trial in s
dtbin = 0.02; % specify the desired bin size for discretization in s
No_Orient = size(spike_times_all_ori,2);

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

% Selected spiking observations
ysamp_init = ysamp_all(:,:,selected_orient);

L_test = 5; % Number of trials used for testing
L_train = size(ysamp_init,2) - L_test; % Number of trials used for training

Indices_test = zeros(J,L_test); % Selected trials for testing
Indices_train = zeros(J,L_train); % Selected trials for training
ysamp_train = zeros(size(ysamp_init,1),L_train,J); % Training set
ysamp_test = zeros(size(ysamp_init,1),L_test,J); % Testing set

% Extract the training and testing test for each stimuli
for j = 1:J
    temp = randperm(50,L_test);
    Indices_test(j,:) = temp;
    Indices_train(j,:) = setdiff(1:50,temp);
    ysamp_train(:,:,j) = ysamp_init(:,squeeze(Indices_train(j,:)),j);
    ysamp_test(:,:,j) = ysamp_init(:,squeeze(Indices_test(j,:)),j);
end

% Visualize sample descretized spike counts of the training set
figure(1)
for l = 1:5
    for j = 1:J
        subplot(5,J,J*(l-1)+j)
        stem(xsamp,ysamp_train(:,l,j),'k');
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

%% Step 2: Proposed CMP-EPL model: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method

cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
min_change_in_hyp = 5*10^(-3); % the threshold for checking convergence of hyper-parameters
tolerance_lambda_init = 10^(-2); % the threshold for checking onvergence of estimated firing rate

tic
% Optimize for the hyper-parameters using the selected optimization method on training data
estimated_hyperparameters_CMP_EPL = optimize_hyperparameters_multi_stimuli_CMP_EPL(ysamp_train,xsamp,dtbin,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda_init);
toc

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
fprintf('noise len:  %7.2f \n', estimated_hyperparameters_CMP_EPL.noise.len);
fprintf('noise rho:  %7.2f \n', estimated_hyperparameters_CMP_EPL.noise.rho);
fprintf('noise q:  %7.2f \n', estimated_hyperparameters_CMP_EPL.noise.q);
for j = 1:J
    fprintf('Stimulus %d signal len:  %7.2f \n', j, estimated_hyperparameters_CMP_EPL.signal.len(j));
    fprintf('Stimulus %d signal rho:  %7.2f \n', j, estimated_hyperparameters_CMP_EPL.signal.rho(j));
end

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_q_final_CMP_EPL = [estimated_hyperparameters_CMP_EPL.noise.rho,estimated_hyperparameters_CMP_EPL.noise.len,estimated_hyperparameters_CMP_EPL.noise.q];
signal_rho_len_final_CMP_EPL = zeros(J,2);
signal_rho_len_final_CMP_EPL(:,1) = estimated_hyperparameters_CMP_EPL.signal.rho;
signal_rho_len_final_CMP_EPL(:,2) = estimated_hyperparameters_CMP_EPL.signal.len;

% Evaluate the final latent estimates of testing data
[ELBO_final_CMP_EPL_test,Noise_latents_CMP_EPL_test,Signal_latents_CMP_EPL_test,Firing_rate_per_bin_CMP_EPL_test,Noise_covariance_CMP_EPL_test] = derive_ELBO_VI_multi_stimuli_CMP_EPL(signal_rho_len_final_CMP_EPL,noise_rho_len_q_final_CMP_EPL,xsamp,ysamp_test,dtbin,tolerance_lambda_final,use_log);

% Evaluate the final signal latent estimates of training data, for cross validated likelood derivation
[~,~,Signal_latents_CMP_EPL_train,~] = derive_ELBO_VI_multi_stimuli_CMP_EPL(signal_rho_len_final_CMP_EPL,noise_rho_len_q_final_CMP_EPL,xsamp,ysamp_train,dtbin,tolerance_lambda_final,use_log);

%% Step 3: CMP-RBF model: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method

cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
min_change_in_hyp = 5*10^(-3); % the threshold for checking convergence of hyper-parameters
tolerance_lambda_init = 10^(-2); % the threshold for checking onvergence of estimated firing rate

tic
% Optimize for the hyper-parameters using the selected optimization method
estimated_hyperparameters_CMP_RBF = optimize_hyperparameters_multi_stimuli_CMP_RBF(ysamp_train,xsamp,dtbin,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda_init);
toc

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
fprintf('noise len:  %7.2f \n', estimated_hyperparameters_CMP_RBF.noise.len);
fprintf('noise rho:  %7.2f \n', estimated_hyperparameters_CMP_RBF.noise.rho);
for j = 1:J
    fprintf('Stimulus %d signal len:  %7.2f \n', j, estimated_hyperparameters_CMP_RBF.signal.len(j));
    fprintf('Stimulus %d signal rho:  %7.2f \n', j, estimated_hyperparameters_CMP_RBF.signal.rho(j));
end

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_final_CMP_RBF = [estimated_hyperparameters_CMP_RBF.noise.rho,estimated_hyperparameters_CMP_RBF.noise.len];

signal_rho_len_final_CMP_RBF = zeros(J,2);
signal_rho_len_final_CMP_RBF(:,1) = estimated_hyperparameters_CMP_RBF.signal.rho;
signal_rho_len_final_CMP_RBF(:,2) = estimated_hyperparameters_CMP_RBF.signal.len;

% Evaluate the final latent estimates of testing data
[ELBO_final_CMP_RBF_test, Noise_latents_CMP_RBF_test, Signal_latents_CMP_RBF_test, Firing_rate_per_bin_CMP_RBF_test, Noise_covariance_CMP_RBF_test] = derive_ELBO_VI_multi_stimuli_CMP_RBF(signal_rho_len_final_CMP_RBF,noise_rho_len_final_CMP_RBF,xsamp,ysamp_test,dtbin,tolerance_lambda_final,use_log);

% Evaluate the final signal latent estimates of training data, for cross validated likelood derivation
[~,~,Signal_latents_CMP_RBF_train,~] = derive_ELBO_VI_multi_stimuli_CMP_RBF(signal_rho_len_final_CMP_RBF,noise_rho_len_final_CMP_RBF,xsamp,ysamp_train,dtbin,tolerance_lambda_final,use_log);

%% Step 4: CMP-Matern-0: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

Matern_order = 0;
optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method

cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
min_change_in_hyp = 5*10^(-3); % the threshold for checking convergence of hyper-parameters
tolerance_lambda_init = 10^(-2); % the threshold for checking onvergence of estimated firing rate

tic
% Optimize for the hyper-parameters using the selected optimization method on training data
estimated_hyperparameters_CMP_Matern_0 = optimize_hyperparameters_multi_stimuli_CMP_Matern(ysamp_train,xsamp,dtbin,Matern_order,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda_init);
toc

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
fprintf('noise len:  %7.2f \n', estimated_hyperparameters_CMP_Matern_0.noise.len);
fprintf('noise rho:  %7.2f \n', estimated_hyperparameters_CMP_Matern_0.noise.rho);
for j = 1:J
    fprintf('Stimulus %d signal len:  %7.2f \n', j, estimated_hyperparameters_CMP_Matern_0.signal.len(j));
    fprintf('Stimulus %d signal rho:  %7.2f \n', j, estimated_hyperparameters_CMP_Matern_0.signal.rho(j));
end

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_final_CMP_Matern_0 = [estimated_hyperparameters_CMP_Matern_0.noise.rho,estimated_hyperparameters_CMP_Matern_0.noise.len];
signal_rho_len_final_CMP_Matern_0 = zeros(J,2);
signal_rho_len_final_CMP_Matern_0(:,1) = estimated_hyperparameters_CMP_Matern_0.signal.rho;
signal_rho_len_final_CMP_Matern_0(:,2) = estimated_hyperparameters_CMP_Matern_0.signal.len;

% Evaluate the final latent estimates of testing data
[ELBO_final_CMP_Matern_0_test, Noise_latents_CMP_Matern_0_test, Signal_latents_CMP_Matern_0_test, Firing_rate_per_bin_CMP_Matern_0_test, Noise_covariance_CMP_Matern_0_test] = derive_ELBO_VI_multi_stimuli_CMP_Matern(signal_rho_len_final_CMP_Matern_0,noise_rho_len_final_CMP_Matern_0,xsamp,ysamp_test,dtbin,tolerance_lambda_final,use_log,Matern_order);

% Evaluate the final signal latent estimates of training data, for cross validated likelood derivation
[~,~,Signal_latents_CMP_Matern_0_train,~] = derive_ELBO_VI_multi_stimuli_CMP_Matern(signal_rho_len_final_CMP_Matern_0,noise_rho_len_final_CMP_Matern_0,xsamp,ysamp_train,dtbin,tolerance_lambda_final,use_log,Matern_order);

%% Step 5: CMP-Matern-1: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

Matern_order = 1;
optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method

cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
min_change_in_hyp = 5*10^(-3); % the threshold for checking convergence of hyper-parameters
tolerance_lambda_init = 10^(-2); % the threshold for checking onvergence of estimated firing rate

tic
% Optimize for the hyper-parameters using the selected optimization method on training data
estimated_hyperparameters_CMP_Matern_1 = optimize_hyperparameters_multi_stimuli_CMP_Matern(ysamp_train,xsamp,dtbin,Matern_order,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda_init);
toc

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
fprintf('noise len:  %7.2f \n', estimated_hyperparameters_CMP_Matern_1.noise.len);
fprintf('noise rho:  %7.2f \n', estimated_hyperparameters_CMP_Matern_1.noise.rho);
for j = 1:J
    fprintf('Stimulus %d signal len:  %7.2f \n', j, estimated_hyperparameters_CMP_Matern_1.signal.len(j));
    fprintf('Stimulus %d signal rho:  %7.2f \n', j, estimated_hyperparameters_CMP_Matern_1.signal.rho(j));
end

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

% Vectorise the hyper-parameters
noise_rho_len_final_CMP_Matern_1 = [estimated_hyperparameters_CMP_Matern_1.noise.rho,estimated_hyperparameters_CMP_Matern_1.noise.len];
signal_rho_len_final_CMP_Matern_1 = zeros(J,2);
signal_rho_len_final_CMP_Matern_1(:,1) = estimated_hyperparameters_CMP_Matern_1.signal.rho;
signal_rho_len_final_CMP_Matern_1(:,2) = estimated_hyperparameters_CMP_Matern_1.signal.len;

% Evaluate the final latent estimates of testing data
[ELBO_final_CMP_Matern_1_test, Noise_latents_CMP_Matern_1_test, Signal_latents_CMP_Matern_1_test, Firing_rate_per_bin_CMP_Matern_1_test,Noise_covariance_CMP_Matern_1_test] = derive_ELBO_VI_multi_stimuli_CMP_Matern(signal_rho_len_final_CMP_Matern_1,noise_rho_len_final_CMP_Matern_1,xsamp,ysamp_test,dtbin,tolerance_lambda_final,use_log,Matern_order);

% Evaluate the final signal latent estimates of training data, for cross validated likelood derivation
[~,~,Signal_latents_CMP_Matern_1_train,~] = derive_ELBO_VI_multi_stimuli_CMP_Matern(signal_rho_len_final_CMP_Matern_1,noise_rho_len_final_CMP_Matern_1,xsamp,ysamp_train,dtbin,tolerance_lambda_final,use_log,Matern_order);

%% Step 4: CMP-EPL Independent Noise Model: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method

cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
min_change_in_hyp = 5*10^(-3); % the threshold for checking convergence of hyper-parameters
tolerance_lambda_init = 10^(-2); % the threshold for checking onvergence of estimated firing rate
    
tic
for j = 1:J
    fprintf('\n Estimating hyperparameters for Stimulus %d \n', j)
    % Optimize for the hyper-parameters using the selected optimization method
    estimated_hyperparameters = optimize_hyperparameters_CMP_EPL(squeeze(ysamp_train(:,:,j)),xsamp,dtbin,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda_init);    
    estimated_hyperparameters_CMP_EPL_independent_noise{j} = estimated_hyperparameters;
end
toc

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
for j = 1:J
    estimated_hyperparameters = estimated_hyperparameters_CMP_EPL_independent_noise{j};
    fprintf('Stimulus %d noise len:  %7.2f \n', j, estimated_hyperparameters.noise.len);
    fprintf('Stimulus %d noise rho:  %7.2f \n', j, estimated_hyperparameters.noise.rho);
    fprintf('Stimulus %d noise q:  %7.2f \n', j, estimated_hyperparameters.noise.q);
    fprintf('Stimulus %d signal len:  %7.2f \n', j, estimated_hyperparameters.signal.len);
    fprintf('Stimulus %d signal rho:  %7.2f \n', j, estimated_hyperparameters.signal.rho);
end

tolerance_lambda_final = 10^(-5); % Use a finer tolerence for the final estimate to improve the accuracy of the latent estimates
use_log = false; % No need to convert to log-domain for evaluation of latents

ELBO_final_CMP_EPL_independent_noise_test = zeros(1,J);
Noise_latents_CMP_EPL_independent_noise_test = ysamp_test*0;
Signal_latents_CMP_EPL_independent_noise_test = zeros(size(ysamp_test,1),size(ysamp_test,3));
Firing_rate_per_bin_CMP_EPL_independent_noise_test = ysamp_test*0;
Noise_covariance_CMP_EPL_independent_noise_test = zeros(size(ysamp_test,1),size(ysamp_test,1),size(ysamp_test,2),size(ysamp_test,3));
% Find the signal latents of training data for cross validation
Signal_latents_CMP_EPL_independent_noise_train = zeros(size(ysamp_test,1),size(ysamp_test,3));

for j = 1:J   
    estimated_hyperparameters = estimated_hyperparameters_CMP_EPL_independent_noise{j};
    % Vectorise the hyper-parameters
    noise_rho_len_q_final_CMP_EPL_independent_noise = [estimated_hyperparameters.noise.rho,estimated_hyperparameters.noise.len,estimated_hyperparameters.noise.q];
    signal_rho_len_final_CMP_EPL_independent_noise = [estimated_hyperparameters.signal.rho, estimated_hyperparameters.signal.len];
    % Evaluate the final latent estimates of test data
    [ELBO_final,Noise_latents,Signal_latents,Firing_rate_per_bin,Noise_covariance] = derive_ELBO_VI_CMP_EPL(signal_rho_len_final_CMP_EPL_independent_noise,noise_rho_len_q_final_CMP_EPL_independent_noise,xsamp,squeeze(ysamp_test(:,:,j)),dtbin,tolerance_lambda_final,use_log);
    ELBO_final_CMP_EPL_independent_noise_test(1,j) = ELBO_final;
    Noise_latents_CMP_EPL_independent_noise_test(:,:,j) = Noise_latents;
    Signal_latents_CMP_EPL_independent_noise_test(:,j) = Signal_latents;
    Firing_rate_per_bin_CMP_EPL_independent_noise_test(:,:,j) = Firing_rate_per_bin;
    Noise_covariance_CMP_EPL_independent_noise_test(:,:,:,j) = Noise_covariance;
    % Evaluate the final signal latent estimates of training data
    [~,~,Signal_latents,~] = derive_ELBO_VI_CMP_EPL(signal_rho_len_final_CMP_EPL_independent_noise,noise_rho_len_q_final_CMP_EPL_independent_noise,xsamp,squeeze(ysamp_train(:,:,j)),dtbin,tolerance_lambda_final,use_log);
    Signal_latents_CMP_EPL_independent_noise_train(:,j) = Signal_latents;
end

%% Step 5: Poisson-GP (No Noise) Model: Perform Variational Inference to extract the underlying Signal and Noise Gaussian Processes

optimization_method = 'fmincon'; % set this to 'fmincon' for constrained optimization, 'fminunc' to gradient based method, or 'fminsearch' for simplex method
    
tic
for j = 1:J
    fprintf('\n Estimating hyperparameters for Stimulus %d \n', j)
    % Optimize for the hyper-parameters using the selected optimization method
    estimated_hyperparameters = optimize_hyperparameters_Poisson_GP(squeeze(ysamp_train(:,:,j)),xsamp,dtbin,optimization_method);    
    estimated_hyperparameters_Poisson_GP{j} = estimated_hyperparameters;
end
toc

fprintf('\nLearned hyperparameters:\n----------------------------------\n');
for j = 1:J
    estimated_hyperparameters = estimated_hyperparameters_Poisson_GP{j};
    fprintf('Stimulus %d signal len:  %7.2f \n', j, estimated_hyperparameters.signal.len);
    fprintf('Stimulus %d signal rho:  %7.2f \n', j, estimated_hyperparameters.signal.rho);
end
    
use_log = false; % No need to convert to log-domain for evaluation of latents

ELBO_final_Poisson_GP_test = zeros(1,J);
Signal_latents_Poisson_GP_test = zeros(size(ysamp_test,1),size(ysamp_test,3));
Firing_rate_per_bin_Poisson_GP_test = ysamp_test*0;
Signal_latents_Poisson_GP_train = zeros(size(ysamp_test,1),size(ysamp_test,3));

for j = 1:J
    estimated_hyperparameters = estimated_hyperparameters_Poisson_GP{j};
    % Vectorise the hyper-parameters
    signal_rho_len_final_Poisson_GP = [estimated_hyperparameters.signal.rho, estimated_hyperparameters.signal.len];
    % Evaluate the final latent estimates of testing data
    [ELBO_final,Signal_latents,Firing_rate_per_bin] = derive_ELBO_VI_Poisson_GP(signal_rho_len_final_Poisson_GP,xsamp,squeeze(ysamp_test(:,:,j)),dtbin,use_log);
    ELBO_final_Poisson_GP_test(1,j) = ELBO_final;
    Signal_latents_Poisson_GP_test(:,j) = Signal_latents;
    Firing_rate_per_bin_Poisson_GP_test(:,:,j) = Firing_rate_per_bin;
    % Evaluate the final latent estimates of training data
    [~,Signal_latents,~] = derive_ELBO_VI_Poisson_GP(signal_rho_len_final_Poisson_GP,xsamp,squeeze(ysamp_train(:,:,j)),dtbin,use_log);
    Signal_latents_Poisson_GP_train(:,j) = Signal_latents;   
end

%% Step 6: Plot the inferred latents on test data using each method for visual comparison

linewidth = 1.5;

figure(2)
L_plot = 1;
L_start = 0;
for j = 1:J
    for l = 1:L_plot
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 0*J*L_plot)
        stem(xsamp,ysamp_test(:,l+L_start,j),'k')
        title(['Stimulus ' num2str(j)])
        if j == 1
            ylabel('Spike counts: y_t','FontSize', 10)
        end
        axis([min(xsamp), max(xsamp),0,max(2,max(ysamp_test(:,l+L_start,j)))])
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 1*J*L_plot)
        plot(xsamp,Firing_rate_per_bin_CMP_EPL_test(:,l+L_start,j)/dtbin,'r', 'linewidth', linewidth)
        hold on;
        plot(xsamp,Firing_rate_per_bin_CMP_RBF_test(:,l+L_start,j)/dtbin,'m', 'linewidth', linewidth)
        hold on;
        plot(xsamp,Firing_rate_per_bin_CMP_EPL_independent_noise_test(:,l+L_start,j)/dtbin,'--', 'Color', [0.5,0,0], 'linewidth', linewidth)
        hold on;
        plot(xsamp,Firing_rate_per_bin_Poisson_GP_test(:,l+L_start,j)/dtbin,'g', 'linewidth', linewidth)
        hold off;
        if j == 1
            ylabel('Firing rate (Hz)','FontSize', 10)
            legend('CMP-EPL','CMP-RBF','CMP-EPL independent noise hyp','Poisson-GP')
        end
        axis([min(xsamp), max(xsamp),0,100])
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 2*J*L_plot)
        plot(xsamp,exp(Signal_latents_CMP_EPL_test(:,j)),'r', 'linewidth', linewidth)
        hold on;
        plot(xsamp,exp(Signal_latents_CMP_RBF_test(:,j)),'m', 'linewidth', linewidth)
        hold on;
        plot(xsamp,exp(Signal_latents_CMP_EPL_independent_noise_test(:,j)),'--', 'Color', [0.5,0,0], 'linewidth', linewidth)
        hold on;
        plot(xsamp,exp(Signal_latents_Poisson_GP_test(:,j)),'g', 'linewidth', linewidth)        
        hold off;
        if j == 1
            ylabel('Estimated Signal Latents','FontSize', 10)
        end
        axis([min(xsamp), max(xsamp),0,50])
        subplot(4,J*L_plot,(j-1)*L_plot + l+ 3*J*L_plot)
        plot(xsamp,exp(Noise_latents_CMP_EPL_test(:,l+L_start,j)),'r', 'linewidth', linewidth)
        hold on;
        plot(xsamp,exp(Noise_latents_CMP_RBF_test(:,l+L_start,j)),'m', 'linewidth', linewidth)
        hold on;
        plot(xsamp,exp(Noise_latents_CMP_EPL_independent_noise_test(:,l+L_start,j)),'--', 'Color', [0.5,0,0], 'linewidth', linewidth)
        hold off;
        if j == 1
            ylabel('Estimated Noise Latents','FontSize', 10)
        end      
        axis([min(xsamp), max(xsamp),0,50])
    end
end

%% Step 7: Find the log likelihood of each CMP based model on test data using Importance Sampling (IS) and compare with the Poisson Signal GP model

MC_interations = 1000; % Specify the number of Monte Carlo iterations
no_log_likelihood_repeats = 1; % Specify the number of repeats of the IS procedure, if you need to derive confidence bounds

% First, derive the log likelihood of the baseline Poisson model (no noise)
log_likelihood_Poisson_GP_test = derive_log_likelihood_importance_sampling(ysamp_test,Signal_latents_Poisson_GP_train,dtbin);

% Next, derive the log likelihood of the other models using IS
log_likelihood_CMP_EPL_test = zeros(no_log_likelihood_repeats,1);
log_likelihood_CMP_RBF_test = zeros(no_log_likelihood_repeats,1);
log_likelihood_CMP_Matern_0_test = zeros(no_log_likelihood_repeats,1);
log_likelihood_CMP_Matern_1_test = zeros(no_log_likelihood_repeats,1);
log_likelihood_CMP_EPL_independent_noise_test = zeros(no_log_likelihood_repeats,1);

for repeat = 1:no_log_likelihood_repeats
    log_likelihood_CMP_EPL_test(repeat) = derive_log_likelihood_importance_sampling(ysamp_test,Signal_latents_CMP_EPL_train,dtbin,xsamp,Noise_latents_CMP_EPL_test,Noise_covariance_CMP_EPL_test,estimated_hyperparameters_CMP_EPL,MC_interations);
    log_likelihood_CMP_RBF_test(repeat) = derive_log_likelihood_importance_sampling(ysamp_test,Signal_latents_CMP_RBF_train,dtbin,xsamp,Noise_latents_CMP_RBF_test,Noise_covariance_CMP_RBF_test,estimated_hyperparameters_CMP_RBF,MC_interations);
    log_likelihood_CMP_Matern_0_test(repeat) = derive_log_likelihood_importance_sampling(ysamp_test,Signal_latents_CMP_Matern_0_train,dtbin,xsamp,Noise_latents_CMP_Matern_0_test,Noise_covariance_CMP_Matern_0_test,estimated_hyperparameters_CMP_Matern_0,MC_interations,0);   
    log_likelihood_CMP_Matern_1_test(repeat) = derive_log_likelihood_importance_sampling(ysamp_test,Signal_latents_CMP_Matern_1_train,dtbin,xsamp,Noise_latents_CMP_Matern_1_test,Noise_covariance_CMP_Matern_1_test,estimated_hyperparameters_CMP_Matern_1,MC_interations,1);   
    log_likelihood_CMP_EPL_independent_noise_test(repeat) = derive_log_likelihood_importance_sampling(ysamp_test,Signal_latents_CMP_EPL_independent_noise_train,dtbin,xsamp,Noise_latents_CMP_EPL_independent_noise_test,Noise_covariance_CMP_EPL_independent_noise_test,estimated_hyperparameters_CMP_EPL_independent_noise,MC_interations);   
end

% Plot the difference of the LL of each model with respect to the Poisson Signal GP model
scaling_fac = (log(2)*J*L_test);
figure(3)
c = bar([log_likelihood_CMP_EPL_test/scaling_fac-log_likelihood_Poisson_GP_test/scaling_fac,log_likelihood_CMP_RBF_test/scaling_fac-log_likelihood_Poisson_GP_test/scaling_fac,log_likelihood_CMP_Matern_0_test/scaling_fac-log_likelihood_Poisson_GP_test/scaling_fac,log_likelihood_CMP_Matern_1_test/scaling_fac-log_likelihood_Poisson_GP_test/scaling_fac,log_likelihood_CMP_EPL_independent_noise_test/scaling_fac-log_likelihood_Poisson_GP_test/scaling_fac]);
c.FaceColor = 'flat';
c.CData(1,:) = [1 0 0];
c.CData(2,:) = [1 0 1];
c.CData(3,:) = [0 0.5 0];
c.CData(4,:) = [0 1 0];
c.CData(5,:) = [0.5 0 0];
somenames={'Exp power'; 'RBF';'Matern 1/2';'Matern 3/2';'Independent noise'};
set(gca,'xticklabel',somenames,'FontSize', 12)
title('log likelihood of held out trials - difference wrt Poisson Signal GP','FontSize', 14)
%% Step 8: Goris independent noise model: derive the log-likelihood of test data

log_likelihood_Goris_ind_per_stimulus_test = zeros(J,1);
for j = 1:J
    ysamp_train_j = squeeze(ysamp_train(:,:,j));
    ysamp_test_j = squeeze(ysamp_test(:,:,j));
    fprintf('\n Estimating hyperparameters for Stimulus %d \n', j)
    % Optimize for the hyper-parameters by maximizing the likelihood
    [r,s] = optimize_hyp_Goris_independent_model(ysamp_train_j);
    estimated_hyperparameters.r = r;
    estimated_hyperparameters.s = s;
    estimated_hyperparameters_Goris_ind{j} = estimated_hyperparameters;
    log_likelihood_Goris_ind_per_stimulus_test(j) = -1*derive_neg_log_like_Goris_independent_model(r,s,ysamp_test_j);
end
% Get the final log-likelihood by summing across stimuli
log_likelihood_Goris_ind_test = sum(log_likelihood_Goris_ind_per_stimulus_test);

%% Step 9: Goris constant noise model: derive the log-likelihood of test data

% Derive the hyper-parameter settings of the Gamma prior of the baseline Poisson model
mean_rate = squeeze(mean(ysamp_train,2));
mean_all = mean(mean_rate(:));
variance_all = var(mean_rate(:));
beta = mean_all/variance_all;
alpha = max(mean_all * beta,1) + 10^(-2);

log_likelihood_Goris_constant_test = 0;
options = optimoptions('fmincon','StepTolerance',1e-3,'OptimalityTolerance',1e-3,'MaxFunctionEvaluations',100,'Display','off');

for j = 1:J
    % get the mean firing rate from the baseline Poisson model
    lambda = (squeeze(sum(squeeze(ysamp_train(:,:,j)),2)) + alpha - 1)/(beta + L_train);
    % estimate hyper-parameters of the Gamma distribution using the baseline Goris model
    estimate_hyp = @(hyp)eval_neg_log_like_Goris_model(hyp,squeeze(sum(squeeze(ysamp_train(:,:,j)),1)));
    [estimated_hyp] =  fmincon(estimate_hyp,[1,1],[],[],[],[],[0,0],[],[],options);
    r_0 = estimated_hyp(1);
    % derive the test log-likelihood using Monte Carlo sampling
    for l = 1:L_test
        ysamp = squeeze(squeeze(ysamp_test(:,l,j)));
        log_likelihood_Goris_constant_test = log_likelihood_Goris_constant_test + get_test_LL_Goris_constant_noise_model(ysamp, lambda, r_0, MC_interations);
    end
end

%% Step 10: Baseline Poisson model: derive the log-likelihood of test data

% Derive the hyper-parameter settings of the Gamma prior of the baseline Poisson model

mean_rate = squeeze(mean(ysamp_train,2));
mean_all = mean(mean_rate(:));
variance_all = var(mean_rate(:));
beta = mean_all/variance_all;
alpha = max(mean_all * beta,1) + 10^(-2);

log_likelihood_baseline_Poisson_test = 0;
for j = 1:J
    % get the mean firing rate from the baseline Poisson model
    lambda = (squeeze(sum(squeeze(ysamp_train(:,:,j)),2)) + alpha - 1)/(beta + L_train);
    % derive the test log-likelihood for each test trial
    for l = 1:L_test
        ysamp = squeeze(squeeze(ysamp_test(:,l,j)));
        log_likelihood_baseline_Poisson_test = log_likelihood_baseline_Poisson_test + sum(-lambda + log(lambda.^ysamp) - log(factorial(ysamp)));
    end
end

%% Step 10: Plot the comparison of the log-likelihood gain of all models relative to the baseline Poisson model

scaling_fac = (log(2)*J*L_test); % to derive the log-likelihood in bits per trial

figure(4)
c = bar([log_likelihood_CMP_EPL_test/scaling_fac-log_likelihood_baseline_Poisson_test/scaling_fac,log_likelihood_CMP_RBF_test/scaling_fac-log_likelihood_baseline_Poisson_test/scaling_fac,log_likelihood_CMP_Matern_0_test/scaling_fac-log_likelihood_baseline_Poisson_test/scaling_fac,log_likelihood_CMP_Matern_1_test/scaling_fac-log_likelihood_baseline_Poisson_test/scaling_fac, log_likelihood_Poisson_GP_test/scaling_fac - log_likelihood_baseline_Poisson_test/scaling_fac, log_likelihood_Goris_ind_test/scaling_fac-log_likelihood_baseline_Poisson_test/scaling_fac, log_likelihood_Goris_constant_test/scaling_fac - log_likelihood_baseline_Poisson_test/scaling_fac],0.7);
c.FaceColor = 'flat';
c.CData(1,:) = [1 0 0];
c.CData(2,:) = [1 0 1];
c.CData(3,:) = [0 0.5 0];
c.CData(4,:) = [0 1 0];
c.CData(5,:) = 0.8*[1 1 1];
c.CData(6,:) = [0 0 1];
c.CData(7,:) = [0.5 0.5 1];
somenames={'Exp power (Proposed)'; 'RBF';'Matern 1/2';'Matern 3/2'; 'Poisson Signal GP'; 'Goris indep noise'; 'Goris constant noise'};
set(gca,'xticklabel',somenames,'FontSize', 12)
title(['log likelihood - difference wrt baseline Poisson model: dt = ' num2str(round(1000*dtbin)) ' ms'],'FontSize', 14)


