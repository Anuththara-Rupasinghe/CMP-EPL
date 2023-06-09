function [cdiag,Bfft] = Krbf_fourier(xx,len,rho,fdp)
% Compute basis for RBF covariance in Fourier domain 
%
% [cdiag,Bfft] = Krbf_fourier(xx,len,rho,fdp)
%
% Covariance given by:    K = Bfft * diag(cdiag) * Bfft'
%
% INPUT:
% -----
%           xx [nx x d] - spatial input locations in a d-dimensional space (d<=3)
%           len [1 x 1]  - length scale
%           rho [1 x 1]  - marginal variance
%          fdp (struct) - struct governing Fourier domain representation
%             .circinterval [2 x d] - circular support in each stimulus dimension
%             .condthresh   [1 x 1] - condition number for thresholding for small eigenvalues 
%
% OUTPUT:
% ------
%   cdiag [nb x 1] - vector with thresholded eigenvalues of K
%    Bfft [nb x nx] - column vectors define orthogonal basis for K (on Reals)
%
% --------
% Note: Bfft is really the "inverse fourier transform" matrix.  
% Bfft'*x computes real-valued DFT of x; Bfft*x is inverse real-valued DFT. 
% The naming inconsistency is a bit annoying, but this is the convention in
% the rest of the  Fourier-domain GP code. 


% extract size of input data
nd = size(xx,2);  % number of input dimensions

% compute Fourier-domain basis and frequencies
[Bfft,wwsq] = mkRBFfourierBasis(xx,len,fdp); 

% transform rho constant
trho = transformRho(rho,len,nd,1); 

% compute diagonal elements of prior covariance
cdiag =  trho * exp(-.5*(sum(wwsq,2)*len.^2)); 
