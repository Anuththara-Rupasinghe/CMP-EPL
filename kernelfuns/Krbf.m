function K = Krbf(x,len,rho)
% Make covariance matrix using RBF covariance function (aka
% squared-exponential kernel)
%
% K = Krbf(x,prs)
%
% Covariance matrix parametrized as:
%  K_ij = rho*exp(((x_i-x_j)^2/(2*len^2))
%
% INPUTS:
%     x [nx x d] - points at which to evaluate kernel function
%     len - length scale
%     rho - marginal variance
%
% OUTPUT:
%   K [nx x nx] - covariance matrix


% Compute squared Euclidean distance between all pairs of points in x
sqrdists = sum(bsxfun(@minus,permute(x,[1 3 2]),permute(x,[3 1 2])).^2,3);

% % Old version (required neural networks toolbox):
% sqrdists = dist(x').^2; % squared distances between points

% Compute covariance matrix
K = rho*exp(-.5*sqrdists/len.^2); % the covariance matrix
    