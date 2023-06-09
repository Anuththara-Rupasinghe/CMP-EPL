function K = Kexppow_circular(xvals,len,rho,p,kperiod)
% K = Kexppow_circular(x,len,rho,p,xperiod)
%
% Make covariance using 1-dimensional periodic exponentiated-power-law kernel
% at an arbitrary (gridded or ungridded) set of x locations 'xvals'.
%
% Covariance matrix is given:
%     K_ij = rho * exp(-0.5 * (|x_i-x_j|/len)^p) )
%
% INPUTS:
%     xvals [nx x 1] - points at which to evaluate kernel function 
%                      (should live within a single period)
%        len [1 x 1] - length scale
%        rho [1 x 1] - marginal variance
%      xcirc [1 x 1] - circular interval (e.g., 2*pi)
%
% OUTPUT:
%   K [nx x nx] - covariance matrix
%
% Note: works for gridded and ungridded x values 

KCUTOFF = 1e8; % 1/KCUTOFF determines where to stop wrapping the kernel

% determine # periods based on when kernel falls to rel height 1/KCUTOFF
nperiods = min(100,ceil(((2*log(KCUTOFF))^(1/p))*len/kperiod));

if nperiods > 100
    warning('Kexppow_circular_fourierGridded: kernel tail requires too many periods!');
    fprintf('\n(Wrapping nperiods=100 to compute circular K)\n');
    nperiods  = 100;
end

% function handle for kernel function
kexppowfun = @(x1,x2)(rho*exp(-0.5*abs((x1-x2)/len).^p));

% Initialize covariance matrix
nx = length(xvals); % number of elements in covariance
K = zeros(nx,nx);   % initialization

% Loop over periods to compute contribution
for jj = -nperiods:nperiods
    % Compute contribution to covariance 
    K = K + kexppowfun(xvals(:)+jj*kperiod,xvals(:)'); % the covariance matrix
end
    
