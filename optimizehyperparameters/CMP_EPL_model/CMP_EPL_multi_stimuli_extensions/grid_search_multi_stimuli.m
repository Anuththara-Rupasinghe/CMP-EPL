%% This function performs a grid search to find a suitable intialization for hyper-parameters

% Inputs:   ysamp:  the spike counts (size: time bins * trials * stimuli)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts

% Output: hyperparameter_init: Intialization of hyper-parameters suggested by the grid search

function hyperparameter_init = grid_search_multi_stimuli(xsamp,ysamp,dtbin)
    
    fprintf('\nPerforming grid search:\n----------------------------------\n');

    rho_levels = [0.2,1];
    len_levels = [0.2,1];
    q_levels = [1,1.5];  
    signal_rho_all = [0.5,2];
    signal_len_all = len_levels;
    noise_rho_all = rho_levels;
    noise_len_all = len_levels;    
    noise_q_all = q_levels;
    
    tolerance_lambda = 1*10^(-2);
    use_log = false;
    
    P = size(ysamp,3);
    hyperparameter_init.signal.rho_init = zeros(P,1);
    hyperparameter_init.signal.len_init = zeros(P,1);
    
    % Find approximate signal hyperparameter settings conditioned on noise
    noise_rho_len_q = [.2,1,2];
    for p = 1:P
        fprintf('\nPerforming grid search stimulus %d:\n----------------------------------\n',p);
        ELBO_all = zeros(length(signal_rho_all),length(signal_len_all));
        for i = 1:length(signal_rho_all)
            for m = 1:length(signal_len_all)
                signal_rho_len = [signal_rho_all(i),signal_len_all(m)];                    
                ELBO_all(i,m) = derive_ELBO_VI(signal_rho_len,noise_rho_len_q,xsamp,squeeze(ysamp(:,:,p)),dtbin,tolerance_lambda,use_log);
            end
        end
        [~, ind] = min(ELBO_all(:));
        [i, m] = ind2sub(size(ELBO_all), ind);
        hyperparameter_init.signal.rho_init(p,1) = signal_rho_all(i);
        hyperparameter_init.signal.len_init(p,1) = signal_len_all(m);
    end
    
    fprintf('\nPerforming grid search noise:\n----------------------------------\n');

    % Find approximate noise hyperparameter settings conditioned on signal
    ELBO_all = zeros(length(noise_rho_all),length(noise_len_all),length(noise_q_all));
    signal_rho_len = [hyperparameter_init.signal.rho_init,hyperparameter_init.signal.len_init];
    for j = 1:length(noise_rho_all)
        for n = 1:length(noise_len_all)
            for q = 1:length(noise_q_all)
                noise_rho_len_q = [noise_rho_all(j),noise_len_all(n),noise_q_all(q)];
                ELBO_all(j,n,q) = derive_ELBO_VI_multi_stimuli(signal_rho_len,noise_rho_len_q,xsamp,ysamp,dtbin,tolerance_lambda,use_log);
            end
        end
    end
    [~, ind] = min(ELBO_all(:));
    [j, n, q] = ind2sub(size(ELBO_all), ind);
    
    % Initialization of noise hyper-parameters
    hyperparameter_init.noise.rho_init = noise_rho_all(j);
    hyperparameter_init.noise.len_init = noise_len_all(n);
    hyperparameter_init.noise.q_init = noise_q_all(q);

    fprintf('\n ----------------------------------\n Initializations selected noise rho = %4.2f, noise len = %4.2f and noise q = %4.2f\n',noise_rho_all(j),noise_len_all(n),noise_q_all(q));
    for p = 1:P
        fprintf('Initializations selected Stimulus %d signal len:  %7.2f \n', p, hyperparameter_init.signal.len_init(p,1));
        fprintf('Initializations selected Stimulus %d signal rho:  %7.2f \n', p, hyperparameter_init.signal.rho_init(p,1));
    end
end