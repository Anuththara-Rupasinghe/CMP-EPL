function K = Krbf_circular(x,len,rho,xperiod)
% K = Krbf_circular(x,len,rho,xperiod)
%
% Make covariance using 1-dimensional periodic RBF kernel, aka periodic
% Gaussian kernel.
%
% Covariance matrix parametrized as:
%  K_ij = rho*exp(((x_i-x_j)^2/(2*len^2))  
%
% INPUTS:
%     x [nx x 1] - points at which to evaluate kernel function (assumes all
%                  live within a single period)
%     len - length scale
%     rho - marginal variance
%   xcirc - circular interval
%
% OUTPUT:
%   K [nx x nx] - covariance matrix

% Initialize covariance
K = zeros(length(x));

% determine # periods to use based on keeping dists <= 3 lengthscales
nperiods = ceil(6*len/xperiod);  % wrap at least 6 stdevs of Gaussian

if nperiods > 50
    warning('Krbf_circular: lengthscale >> period');
    fprintf('\nrank(K) approaching 1!\n');
    fprintf('\n(Using nperiods=50 to compute K)\n');
    nperiods  = 50;
end

% loop over periods
for jj = -nperiods:nperiods
    % Compute squared distances for this particular shift of the data
    sqrdists = ((x(:)+jj*xperiod)-x(:)').^2;

    % Compute contribution to covariance 
    K = K + rho*exp(-.5*sqrdists/len.^2); % the covariance matrix
end
    
