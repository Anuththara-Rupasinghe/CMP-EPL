function [transformedkprs,transformedHessian,J] = transformHprs(kprs,nD,direction,Hessian)
% [transformedkprs,transformedHessian,J] = transformHprs(kprs,nD,direction,Hessian)
%
% Function for transforming from marginal variance "rho" to and from
% alternate parametrization "trho", which decouples it from length scale
% in the Fourier domain.
%
% All hyperparams also log-transformed (forward direction) or
% exponentiated (backward direction).
%
% Inputs (forward direction)
%  kprs [3 x 1] - hyperparameters (forward direction):
%         len - length scale  
%         rho - marginal variance for GP prior (forward direction)
%      nsevar - variance of observation noise (optional)
%
%    nD [1 x 1] - number of dimensions of GP (1, 2, or 3)
% direction - direction ("+1" for forward, "-1" for backward).
%
% Outputs (forward direction)
%   transformedkprs [3 x 1] - transformed parameters
% .         Hessian [3 x 3] - transformed Hessian (reverse direction only)
%                 J [3 x 3] - Jacobian J[i,j] = dkprs(j)/dtkprs(i) 
%                             (reverse direction only)

if direction == 1
    % Forward direction (rho to tilde-rho)
    len = kprs(1);
    rho = kprs(2);
    logtrho = log(rho.*((len*sqrt(2*pi)).^nD)); % transformed rho
    transformedkprs = log(kprs);
    transformedkprs(2) = logtrho;
        
elseif direction == -1
    % Reverse direction (from tilde-rho to rho)
    transformedkprs = exp(kprs);
    len = transformedkprs(1);
    logtrho = transformedkprs(2);
    rho = logtrho./((len*sqrt(2*pi)).^nD);
    transformedkprs(2) = rho;

    % For transforming Hessian
    if nargout > 1
        % Jacobian: J[i,j] = dkprs(j)/dtkprs(i)
        J = [1/len, nD/len, 0; 
            0 1/rho, 0
            0 0 1/transformedkprs(3)];
        transformedHessian = J*Hessian*J';
    end
    
end