%% This function optimizes over the hyper-parameters given observations for the RBF noise kernel model

% Inputs:   ysamp:  the spike counts (size: time bins * trials * stimuli)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts
%           optimization_method:    set to fmincon, fminunc or fminsearch
%           cordinate_update_iterations_max:    maximum number of cordinate updates allowed
%           min_change_in_hyp:  the threshold for checking convergence of hyper-parameters by relative difference in updates
%           tolerance_lambda:   the threshold for checking convergence of estimated firing rate

% Outputs:  hyperparameter_final: estimated final hyper-parameters

function hyperparameter_final = optimize_hyperparameters_multi_stimuli_CMP_RBF(ysamp,xsamp,dtbin,optimization_method,cordinate_update_iterations_max,min_change_in_hyp,tolerance_lambda)
    
    %% Specify the default settings of inputs

    if nargin < 7
        tolerance_lambda = 10^(-2); % the threshold for checking convergence of estimated firing rate
        if nargin < 6
            min_change_in_hyp = 10^(-2); % the threshold for checking convergence of hyper-parameters
            if nargin < 5
                cordinate_update_iterations_max = 10; % maximum number of cordinate updates allowed
                if nargin < 4
                    optimization_method = 'fmincon';
                end
            end 
        end
    end
    
    J = size(ysamp,3); % number of different stimuli
        
    %% For each optimization method, define the initial hyper-parameters and other necessary variables 
        
    switch optimization_method
        
        case 'fmincon' 
%             options = optimoptions('fmincon','Display','iter','PlotFcn',@optimplotfval,'StepTolerance',1e-2,'OptimalityTolerance',1e-2,'MaxFunctionEvaluations',100);
            options = optimoptions('fmincon', 'Display','iter', 'StepTolerance',1e-2,'OptimalityTolerance',1e-2,'MaxFunctionEvaluations',100);
            use_log = false; % There's no need to perform inference in the log-domain for the constrained case since we can enforce the hyper-parameters to be positive
            hyperparameter_init = specify_fmincon_par_multi_stimuli_CMP_RBF(J); % Since the domain is constrained in this case, the algorithm converges to the true parameter settings even without a grip search
            
            signal_rho_len_lb = [hyperparameter_init.signal.rho_lb,hyperparameter_init.signal.len_lb]; % Signal hyper-parameter lower bounds
            signal_rho_len_ub = [hyperparameter_init.signal.rho_ub,hyperparameter_init.signal.len_ub]; % Signal upper-parameter lower bounds

            noise_rho_len_lb = [hyperparameter_init.noise.rho_lb,hyperparameter_init.noise.len_lb]; % Noise hyper-parameter lower bounds
            noise_rho_len_ub = [hyperparameter_init.noise.rho_ub,hyperparameter_init.noise.len_ub]; % Noise upper-parameter lower bounds

            % if using the log domain for inference, accordingly transform the bounds too
            if use_log
                signal_rho_len_lb = log(signal_rho_len_lb);
                signal_rho_len_ub = log(signal_rho_len_ub);
                noise_rho_len_lb = log(noise_rho_len_lb);
                noise_rho_len_ub = log(noise_rho_len_ub);
            end
            
        case 'fminunc'
            options = optimoptions(@fminunc,'Display','iter','PlotFcn',@optimplotfval,'StepTolerance',1e-2,'OptimalityTolerance',1e-2,'MaxFunctionEvaluations',100);
            use_log = true; % use the log-domain since the search is unconstrained
            hyperparameter_init = grid_search_multi_stimuli(xsamp,ysamp,dtbin); % use a grid search to find suitable intialization

        case 'fminsearch'
            options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxIter',100,'TolX',1e-2,'TolFun',1e-3,'MaxFunEvals',100);
            use_log = true; % use the log-domain since the search is unconstrained
            hyperparameter_init = grid_search_multi_stimuli(xsamp,ysamp,dtbin); % use a grid search to find suitable intialization
            
        otherwise
            fprintf('Invalid optimization method: please input only fmincon, fminunc or fminsearch!!!')
            return
    end
    
    %% Infer the signal and noise hyper-parameters using cordinate descend
    
    % Initialize hyper-parameters
    signal_rho_len = [hyperparameter_init.signal.rho_init,hyperparameter_init.signal.len_init];
    noise_rho_len = [hyperparameter_init.noise.rho_init,hyperparameter_init.noise.len_init];
            
    if use_log
        signal_rho_len = log(signal_rho_len);
        noise_rho_len = log(noise_rho_len);
    end
    
    % Store the hyper-parameter updates in each iteration of cordinate descend
    noise_rho_len_all = zeros(cordinate_update_iterations_max,2);
    signal_rho_len_all = zeros(cordinate_update_iterations_max,2,J);
    noise_rho_len_all(1,:) = noise_rho_len;
    signal_rho_len_all(1,:,:) = signal_rho_len';

    % Initialize convergence criteria
    iterations_hyp = 1;
    change_in_hyp = 1;
        
    % Perform the cordinate descend
    while (iterations_hyp <= cordinate_update_iterations_max)&&(change_in_hyp > min_change_in_hyp)
        
        fprintf('\n\nHyper-parameter updating interation %d \n', iterations_hyp);
        
        % Estimate the signal hyper-parameters conditioned on noise hyper-parameters
        fprintf('\nEstimating signal hyper-parameters:\n---------------------------------------------------\n');
        
        for j = 1:J
            fprintf('\n\n Estimating signal hyper-parameters for stimulus %d \n---------------------------------------------------\n', j);
            signal_rho_len_j = squeeze(signal_rho_len(j,:));
            % Define the function for updating signal hyper-parameters 
            estimate_signal = @(signal_rho_len)derive_ELBO_VI_CMP_RBF(signal_rho_len,noise_rho_len,xsamp,squeeze(ysamp(:,:,j)),dtbin,tolerance_lambda,use_log);  

            % Update signal hyper-parameters
            switch optimization_method
                case 'fmincon'
                    [signal_rho_len_j,~,~,~] = fmincon(estimate_signal,signal_rho_len_j,[],[],[],[],signal_rho_len_lb,signal_rho_len_ub,[],options);
                case 'fminunc'
                    [signal_rho_len_j,~,~,~] = fminunc(estimate_signal,signal_rho_len_j,options);
                case 'fminsearch'
                    if iterations_hyp > 2
                        options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxIter',100,'TolX',1e-2,'TolFun',1e-1,'MaxFunEvals',100); 
                    end
                    [signal_rho_len_j,~,~,~] = fminsearch(estimate_signal,signal_rho_len_j,options);
            end
            signal_rho_len(j,:) = signal_rho_len_j;
        end

        % Estimate the noise hyper-parameters conditioned on signal hyper-parameters
        fprintf('\nEstimating noise hyper-parameters:\n----------------------------------------------------\n');
        
        % Define the function for updating the noise hyper-parameters
        estimate_noise = @(noise_rho_len)derive_ELBO_VI_multi_stimuli_CMP_RBF(signal_rho_len,noise_rho_len,xsamp,ysamp,dtbin,tolerance_lambda,use_log);  

        % Update noise hyper-parameters
        switch optimization_method
            case 'fmincon'
                [noise_rho_len,~,~,~] = fmincon(estimate_noise,noise_rho_len,[],[],[],[],noise_rho_len_lb, noise_rho_len_ub, [],options);
            case 'fminunc'
                [noise_rho_len,~,~,~] = fminunc(estimate_noise,noise_rho_len,options);
            case 'fminsearch'
                if iterations_hyp > 2
                    options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxIter',100,'TolX',1e-2,'TolFun',1e-1,'MaxFunEvals',100); 
                end
                [noise_rho_len,~,~,~] = fminsearch(estimate_noise,noise_rho_len,options);
        end
        
        % Store the updated hyper-parameters
        noise_rho_len_all(1+iterations_hyp,:) = noise_rho_len;
        signal_rho_len_all(1+iterations_hyp,:,:) = signal_rho_len';
        
        % Derive the relative change in hyper-paramters to check for convergence
        change_in_hyp_noise = min(abs(noise_rho_len_all(1+iterations_hyp,:) - noise_rho_len_all(iterations_hyp,:))./abs(noise_rho_len_all(1+iterations_hyp,:)));
        change_in_hyp_signal = min(min(abs(signal_rho_len_all(1+iterations_hyp,:,:) - signal_rho_len_all(iterations_hyp,:,:))./abs(signal_rho_len_all(1+iterations_hyp,:,:))));
        change_in_hyp = min(change_in_hyp_noise,change_in_hyp_signal);
        fprintf('\nChange in Hyper-parameters in this interation %f \n', change_in_hyp);
        
        % Update the current interation count
        iterations_hyp = iterations_hyp + 1;

    end
        
    % Re-transform the hyper-parmeters if inferred in log-domain
    if use_log
        signal_rho_len = exp(signal_rho_len);
        noise_rho_len = exp(noise_rho_len);
    end
    
    % The final hyper-parameter estimates
    hyperparameter_final.signal.rho = signal_rho_len(:,1);
    hyperparameter_final.signal.len = signal_rho_len(:,2);
    hyperparameter_final.noise.rho = noise_rho_len(1);
    hyperparameter_final.noise.len = noise_rho_len(2);

% end