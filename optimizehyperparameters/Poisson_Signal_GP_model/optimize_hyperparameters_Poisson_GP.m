%% This function optimizes over the hyper-parameters given observations for the Poisson Signal GP model (no noise component)

% Inputs:   ysamp:  the spike counts (size: time bins * trials)
%           xsamp:  the time axis in seconds (size: time bins * 1)
%           dtbin:  the bin size of spike counts
%           optimization_method:    set to fmincon, fminunc or fminsearch

% Outputs:  hyperparameter_final: estimated final hyper-parameters

function hyperparameter_final = optimize_hyperparameters_Poisson_GP(ysamp,xsamp,dtbin,optimization_method)
    
    %% Specify the default settings of inputs

    if nargin < 4
        optimization_method = 'fmincon';
    end

        
    %% For each optimization method, define the initial hyper-parameters and other necessary variables 
        
    switch optimization_method
        
        case 'fmincon' 
%             options = optimoptions('fmincon','Display','iter','PlotFcn',@optimplotfval,'StepTolerance',1e-2,'OptimalityTolerance',1e-2,'MaxFunctionEvaluations',100);
            options = optimoptions('fmincon','Display', 'iter','StepTolerance',1e-2,'OptimalityTolerance',1e-2,'MaxFunctionEvaluations',100);
            use_log = false; % There's no need to perform inference in the log-domain for the constrained case since we can enforce the hyper-parameters to be positive
            hyperparameter_init = specify_fmincon_par(); % Since the domain is constrained in this case, the algorithm converges to the true parameter settings even without a grip search
            signal_rho_len_lb = [hyperparameter_init.signal.rho_lb,hyperparameter_init.signal.len_lb]; % Signal hyper-parameter lower bounds
            signal_rho_len_ub = [hyperparameter_init.signal.rho_ub,hyperparameter_init.signal.len_ub]; % Signal upper-parameter lower bounds
            % if using the log domain for inference, accordingly transform the bounds too
            if use_log
                signal_rho_len_lb = log(signal_rho_len_lb);
                signal_rho_len_ub = log(signal_rho_len_ub);
            end
            
        case 'fminunc'
            options = optimoptions(@fminunc,'Display','iter','PlotFcn',@optimplotfval,'StepTolerance',1e-3,'OptimalityTolerance',1e-3,'MaxFunctionEvaluations',100);
            use_log = true; % use the log-domain since the search is unconstrained
            hyperparameter_init.signal.rho_init = 1;
            hyperparameter_init.signal.len_init = 0.3;
        case 'fminsearch'
            options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxIter',100,'TolX',1e-2,'TolFun',1e-3,'MaxFunEvals',100);
            use_log = true; % use the log-domain since the search is unconstrained
            hyperparameter_init.signal.rho_init = 1;
            hyperparameter_init.signal.len_init = 0.3;            
        otherwise
            fprintf('Invalid optimization method: please input only fmincon, fminunc or fminsearch!!!')
            return
    end
    
    %% Infer the signal hyper-parameters
    
    % Initialize hyper-parameters
    signal_rho_len = [hyperparameter_init.signal.rho_init,hyperparameter_init.signal.len_init];
    if use_log
        signal_rho_len = log(signal_rho_len);
    end
    
    fprintf('\nEstimating signal hyper-parameters:\n---------------------------------------------------\n');
    
    % Define the function for updating signal hyper-parameters 
    estimate_signal = @(signal_rho_len)derive_ELBO_VI_Poisson_GP(signal_rho_len,xsamp,ysamp,dtbin,use_log);  

    % Update signal hyper-parameters
    switch optimization_method
        case 'fmincon'
            [signal_rho_len,~,~] = fmincon(estimate_signal,signal_rho_len,[],[],[],[],signal_rho_len_lb,signal_rho_len_ub,[],options);
        case 'fminunc'
            [signal_rho_len,~,~] = fminunc(estimate_signal,signal_rho_len,options);
        case 'fminsearch'
            [signal_rho_len,~,~] = fminsearch(estimate_signal,signal_rho_len,options);
    end
    
    % Re-transform the hyper-parmeters if inferred in log-domain
    if use_log
        signal_rho_len = exp(signal_rho_len);
    end
    
    % The final hyper-parameter estimates
    hyperparameter_final.signal.rho = signal_rho_len(1);
    hyperparameter_final.signal.len = signal_rho_len(2);

end