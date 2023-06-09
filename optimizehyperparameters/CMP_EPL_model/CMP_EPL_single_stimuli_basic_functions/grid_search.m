%% This function performs a grid search to find a suitable intialization for hyper-parameters

% Inputs:   ysamp:  the spike counts (size: time bins * trials)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts

% Output: hyperparameter_init: Intialization of hyper-parameters suggested by the grid search

function hyperparameter_init = grid_search(xsamp,ysamp,dtbin)
    
    fprintf('\nPerforming grid search:\n----------------------------------\n');

    rho_levels = [0.2,1];
    len_levels = [0.2,1];
    q_levels = [1,1.5];
    
    signal_rho_all = rho_levels;
    signal_len_all = len_levels;
    noise_rho_all = rho_levels;
    noise_len_all = len_levels;    
    noise_q_all = q_levels;
    
    tolerance_lambda = 1*10^(-2);
    use_log = false;
    
    ELBO_all = zeros(length(signal_rho_all),length(noise_rho_all),length(signal_len_all),length(noise_len_all),length(noise_q_all));
    
    for i = 1:length(signal_rho_all)
        for j = 1:length(noise_rho_all)
            for m = 1:length(signal_len_all)
                for n = 1:length(noise_len_all)
                    for q = 1:length(noise_q_all)
                        fprintf('\n Evaluating ELBO for signal rho = %4.2f, signal len = %4.2f, noise rho = %4.2f, noise len = %4.2f and noise q = %4.2f\n',signal_rho_all(i),signal_len_all(m),noise_rho_all(j),noise_len_all(n),noise_q_all(q));
                        signal_rho_len = [signal_rho_all(i),signal_len_all(m)];                    
                        noise_rho_len_q = [noise_rho_all(j),noise_len_all(n),noise_q_all(q)];
                        ELBO_all(i,j,m,n) = derive_ELBO_VI(signal_rho_len,noise_rho_len_q,xsamp,ysamp,dtbin,tolerance_lambda,use_log);
                    end
                end
            end
        end
    end
    
    [~, ind] = min(ELBO_all(:));
    [i, j, m, n, q] = ind2sub(size(ELBO_all), ind);
    
    % Initialization of signal hyper-parameters
    hyperparameter_init.signal.rho_init = signal_rho_all(i);
    hyperparameter_init.signal.len_init = signal_len_all(m);
    
    % Initialization of noise hyper-parameters
    hyperparameter_init.noise.rho_init = noise_rho_all(j);
    hyperparameter_init.noise.len_init = noise_len_all(n);
    hyperparameter_init.noise.q_init = noise_q_all(q);

    fprintf('\n ----------------------------------\n Initializations selected signal rho = %4.2f, signal len = %4.2f, noise rho = %4.2f, noise len = %4.2f and noise q = %4.2f\n',signal_rho_all(i),signal_len_all(m),noise_rho_all(j),noise_len_all(n),noise_q_all(q));

end