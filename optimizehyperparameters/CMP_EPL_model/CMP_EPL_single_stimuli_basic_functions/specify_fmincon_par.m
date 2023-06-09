%% This function returns the initializations, upper-bounds and lower-bounds of hyper-parameters

function fmincon_par = specify_fmincon_par()

    %% Signal hyper-parameters

    % Initialization of signal hyper-parameters
    fmincon_par.signal.rho_init = 0.1;
    fmincon_par.signal.len_init = 1;
    
    % Upper-bound of signal hyper-parameters
    fmincon_par.signal.rho_ub = 5;
    fmincon_par.signal.len_ub = 6;%10;

    % Lower-bound of signal hyper-parameters
    fmincon_par.signal.rho_lb = 0.05;
    fmincon_par.signal.len_lb = 0.05;
    
    %% Noise hyper-parameters
    
    % Initialization of noise hyper-parameters
    fmincon_par.noise.rho_init = 0.1;
    fmincon_par.noise.len_init = 0.2;%1;
    fmincon_par.noise.q_init = 1;

    % Upper-bound of noise hyper-parameters
    fmincon_par.noise.rho_ub = 1.5;%5;
    fmincon_par.noise.len_ub = 0.5;
    fmincon_par.noise.q_ub = 2;
    
    % Lower-bound of noise hyper-parameters
    fmincon_par.noise.rho_lb = 0.05;
    fmincon_par.noise.len_lb = 0.05;
    fmincon_par.noise.q_lb = 0.6;%0.7;
    
end