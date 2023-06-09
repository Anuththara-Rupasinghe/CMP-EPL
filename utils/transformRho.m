function rhoHat = transformRho(rho,len,nD,direction)
% [rhoHat] = transformRho(rho,len,nD,direction)
%   or
% [rhoHat] = transformRho(rho,len,direction)
%
% Function for transforming from marginal variance rho to alternate
% parametrization "trho", which decoupled rho from dependence on length
% scale in the Fourier domain (forward direction), or back.
%
% Inputs (forward direction)
%         rho - marginal variance for GP prior (forward direction)
%         len - length scale  
%          nD - number of dimensions of GP (1, 2, or 3)
%   direction - direction ("+1" for forward, "-1" for backward).
%
% Outputs (forward direction)
%   rhoHat - transformed rho (or untransformed rho).

if nargin == 3
    % Assume the multiplicity of len determines the nD
    direction = nD; 

    if direction == 1
        % Forward direction (rho to tilde-rho)
        rhoHat = rho*prod((len*sqrt(2*pi))); % transformed rho
        
    elseif direction == -1
        % Reverse direction (from tilde-rho to rho)
        rhoHat = rho/prod(len*sqrt(2*pi)); % marginal variance
        
    else
        error('invalid direction (must be +1 or -1)');
    end

else
   % Use the value of nD
    if direction == 1
        % Forward direction (rho to tilde-rho)
        rhoHat = rho.*((len*sqrt(2*pi)).^nD); % transformed rho
        
    elseif direction == -1
        % Reverse direction (from tilde-rho to rho)
        rhoHat = rho./((len*sqrt(2*pi)).^nD); % marginal variance
        
    else
        error('invalid direction (must be +1 or -1)');
    end

end