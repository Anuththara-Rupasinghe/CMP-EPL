function [cdiag, Bfft] = Kexppow_circular_fourierGridded(xgrid,len,rho,p,kperiod)
% [cdiag, Bfft] = Kexppow_circular_fourierGridded(xvals,len,rho,p,kperiod)
%
% Compute Fourier representation of periodic exponentiated-power-law kernel
% evaluated on a 1D grid of evenly spaced points 'xvals' 
%
% Covariance matrix is given:
%     K_ij = rho * exp(-0.5 * (|x_i-x_j|/len)^p) )
%
% Fourier representation:
%     K = Bfft * diag(cdiag) * Bfft'
%     
%
% INPUTS:
%     xvals [nx x 1] - points at which to evaluate kernel function 
%                      (should live within a single period)
%        len [1 x 1] - length scale
%        rho [1 x 1] - marginal variance
%      xcirc [1 x 1] - circular interval (e.g., 2*pi)
%
% OUTPUT:
%   cdiag [nx x 1] - vector with eigenvalues of K
%    Bfft [nx x nx] - column vectors define orthogonal basis for C (on Reals)
%
% --------
% Note: Bfft is really the "inverse fourier transform" matrix.  
% Bfft'*x computes real-valued DFT of x; Bfft*x is inverse real-valued DFT. 
% The naming inconsistency is a bit annoying, but this is the convention in
% the rest of the  Fourier-domain GP code. 


KCUTOFF = 1e8; % 1/KCUTOFF determines where to stop wrapping the kernel

% determine # periods based on when kernel falls to rel height 1/KCUTOFF
nperiods = min(100,ceil(((2*log(KCUTOFF))^(1/p))*len/kperiod));

if nperiods > 100
    warning('Kexppow_circular_fourierGridded: kernel tail requires too many periods!');
    fprintf('\n(Using nperiods=100 to compute circular K)\n');
    nperiods  = 100;
end

% efficiently compute Fourier spectrum
jjperiod = -nperiods:nperiods;
xgridAllLags = xgrid(:)+(jjperiod*kperiod);  % set of all lags (as a matrix)

% compute single row of K matrix
krow = sum(rho*exp(-0.5*(abs(xgridAllLags)/len).^p),2); 

% compute its Fourier power spectrum
cdiag = abs(fft(krow)); 

if nargout > 1
    % Now make (inverse) FFT basis matrix, if desired
    nx = length(xgrid);
    Bfft = realfftbasis(nx)';
end

% cdiag(1) = [];
% Bfft(:,1) = [];