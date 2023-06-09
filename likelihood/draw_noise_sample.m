%% This function outputs a sample from a Gaussian process (GP)

% Inputs:   xsamp: the support / time axis in seconds (size = time bins * 1) 
%           hyperparameters:the hyper-parameters of the GP 

% Output:   noise_simulated: a simulated sample from the GP

function noise_simulated = draw_noise_sample(xsamp,hyperparameters)
  
    Kmat = hyperparameters.noise.rho*exp(-0.5*abs((xsamp(:)-xsamp(:)')/hyperparameters.noise.len).^hyperparameters.noise.q);

    noise_simulated = mvnrnd(xsamp*0,Kmat);

end