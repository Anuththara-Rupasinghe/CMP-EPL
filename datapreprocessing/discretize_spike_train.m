%% This function returns the discretized spike counts given the spike times

% Inputs:   spike_times:    an array specifying the spike times for each trial
%           T_max:          the duration of each trial in seconds
%           dtbin:          the desired bin size in seconds

% Outputs:  ysamp:          the spike counts (size: time bins * trials)
%           xsamp:          the time axis in seconds (size: time bins * 1)


function [ysamp,xsamp] = discretize_spike_train(spike_times,T_max,dtbin)
    
    K = ceil(T_max/dtbin); % number of time bins per each trial
    t_axis = (dtbin/2:dtbin:K*dtbin)*1000; % the time axis in milliseconds
    xsamp = t_axis'/1000; % time axis in seconds
    L = size(spike_times,2); % number of repeated trials
    
    ysamp = zeros(K,L);
    for l = 1:L
        spike_times_trial = spike_times{l};
        for k = 1:K
            ysamp(k,l) = sum((spike_times_trial > t_axis(k)-1000*dtbin/2)&(spike_times_trial <= t_axis(k) + 1000*dtbin/2));
        end
    end
   
end