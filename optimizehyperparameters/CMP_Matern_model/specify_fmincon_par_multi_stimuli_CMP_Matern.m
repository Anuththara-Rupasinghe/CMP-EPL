%% This function returns the initializations, upper-bounds and lower-bounds of hyper-parameters

function fmincon_par = specify_fmincon_par_multi_stimuli_CMP_Matern(J)

    %% Signal hyper-parameters

    % Initialization of signal hyper-parameters
    fmincon_par.signal.rho_init = 0.1*ones(J,1);
    fmincon_par.signal.len_init = 1*ones(J,1);
    
    % Upper-bound of signal hyper-parameters
    fmincon_par.signal.rho_ub = 5;
    fmincon_par.signal.len_ub = 6;

    % Lower-bound of signal hyper-parameters
    fmincon_par.signal.rho_lb = 0.05;
    fmincon_par.signal.len_lb = 0.05;
    
    %% Noise hyper-parameters
    
    % Initialization of noise hyper-parameters
    fmincon_par.noise.rho_init = 0.1;
    fmincon_par.noise.len_init = 0.2;%1;

    % Upper-bound of noise hyper-parameters
    fmincon_par.noise.rho_ub = 4;
    fmincon_par.noise.len_ub = 0.5;
    
    % Lower-bound of noise hyper-parameters
    fmincon_par.noise.rho_lb = 0.05;%0.09;
    fmincon_par.noise.len_lb = 0.05;%0.09;
    
end