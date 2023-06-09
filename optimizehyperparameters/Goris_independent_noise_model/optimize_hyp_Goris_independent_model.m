%% This function finds the optimal settings of the hyper-parameters for the Goris-independent model

% Inputs:   ysamp:  the spike counts (size: time bins * trials)

% Outputs:  r: shape parameter of the Gamma distribution
%           s: scale parameters of the Gamma distributions at different time points (size: time bins * 1)

function [r,s] = optimize_hyp_Goris_independent_model(ysamp)

cordinate_update_iterations_max = 20;
min_change_in_hyp = 10^(-5);
options = optimoptions('fmincon','StepTolerance',1e-3,'OptimalityTolerance',1e-3,'MaxFunctionEvaluations',100,'Display','off');

% Initialize variables
iterations_hyp = 1;
change_in_hyp = 1;
K = size(ysamp,1);

% Initialize hyper-parameters
s = ones(1,K);
r = 1;

% Store estimated hyper-paramters at each iteratiion
s_all = zeros(cordinate_update_iterations_max,K);
r_all = zeros(cordinate_update_iterations_max,1);
s_all(1,:) = s;
r_all(1) = r;

    % Update the scale and shape hyper-parameters alternatively until convergence
    while (iterations_hyp <= cordinate_update_iterations_max)&&(change_in_hyp > min_change_in_hyp)

        % Estimate the shape hyper-parameters at each time point, conditioned on the scale hyper-parameters
%         fprintf('\nEstimating s hyper-parameters:\n---------------------------------------------------\n');
        for k = 1:K
            estimate_s = @(s_k)derive_neg_log_like_baseline_Goris_model(r,s_k,squeeze(ysamp(k,:)));
            [s_temp] =  fmincon(estimate_s,s(k),[],[],[],[],0,[],[],options);
            s(k) = s_temp;
        end

        % Estimate the scale hyper-parameters for all time points, conditioned on the shape hyper-parameters
%         fprintf('\nEstimating r hyper-parameters:\n---------------------------------------------------\n');
        estimate_r = @(r)derive_neg_log_like_Goris_independent_model(r,s,ysamp);
        r = fmincon(estimate_r,r,[],[],[],[],0,[],[],options);

        % Store the updated hyper-parameters
        s_all(1+iterations_hyp,:) = s;
        r_all(1+iterations_hyp) = r;
        
        % Derive the relative change in hyper-paramters to check for convergence
        change_in_hyp_s = min(abs(s_all(1+iterations_hyp,:) - s_all(iterations_hyp,:))./abs(s_all(1+iterations_hyp,:)));
        change_in_hyp_r = min(min(abs(r_all(1+iterations_hyp) - r_all(iterations_hyp))./abs(r_all(1+iterations_hyp))));
        change_in_hyp = min(change_in_hyp_s,change_in_hyp_r);
%         fprintf('\nChange in Hyper-parameters in this interation %f \n', change_in_hyp);
        
        % Update the current interation count
        iterations_hyp = iterations_hyp + 1;
    end

end