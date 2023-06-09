%% This function derives the negative log-likihood of spike counts given the hyper-parameters for the Goris constant noise model, using Monte Carlo sampling

% Inputs:   ysamp:  the spike counts (size: time bins * trials)
%           lambda: the average firing rate across trials (size: time bins * 1)
%           r_0: the shape parameter of the Gamma distribution
%           MC_iterations:  the number of Monte Carlo iterations

% Outputs:  final_LL: The total negative log-likelihood across all trials and time points

function final_LL = get_test_LL_Goris_constant_noise_model(ysamp, lambda, r_0, MC_iterations)

    s_0 = 1./r_0;
    G_all = gamrnd(r_0,s_0,MC_iterations,1);

    log_likelihood_temp = zeros(1,MC_iterations); % store the LL for all MC iterations

    for i = 1:MC_iterations
        G_i = G_all(i);
        lambda = G_i.*lambda;
        log_likelihood_temp(1,i) = + sum(-lambda + log(lambda.^ysamp) - log(factorial(ysamp)));
    end

    final_LL = logsumexp(log_likelihood_temp,2) - log(MC_iterations);

end