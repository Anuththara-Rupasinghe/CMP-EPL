%% This function returns the confidence bounds given a data distribution

% Inputs:   Data_distribution:      data distribution (no of bin sizes * repeats)
%           bin_size_all:           the bin sizes of the data distribution
%           conf_level:             the desired confidence level in %

% Outputs:  conf_min:               minimum confidence level
%           conf_max:               maximum confidence level
%           conf_bin_size:          bin sizes of the confidence levels

function [conf_min,conf_max,conf_bin_size] = get_conf_bounds(Data_distribution,bin_size_all,conf_level)
    
    % Sort the data distribution in ascending order
    Datadist_sorted = sort(Data_distribution,2,'ascend');

    conf_bin_size = [];
    conf_min = [];
    conf_max = [];
    
    for bin_size = 1:length(bin_size_all)
        temp = Datadist_sorted(bin_size,:);
        % For each bin size, consider only valid entries
        temp = temp(~isnan(temp));
        % Do only if there's more than 1 valid entry
        if length(temp) > 1
            conf_bin_size = [conf_bin_size,bin_size_all(bin_size)];
            high_level = ceil(length(temp)*(1+conf_level)/2);
            low_level = ceil(length(temp)*(1-conf_level)/2);
            conf_min = [conf_min,temp(low_level)];
            conf_max = [conf_max,temp(high_level)];
        end
    end

end