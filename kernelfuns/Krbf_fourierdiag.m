function [cdiag,dcinvdthet,dcdthet,ddcinvdthet] = Krbf_fourierdiag(len,trho,wwsq)
% [cdiag,dcinvdthet,dcdthet,ddcinvdthet] = Krbf_fourierdiag(len,trho,wwsq)
%
% Compute discrete RBF covariance eigenspectrum in Fourier domain, and
% its derivatives w.r.t. to the model parameters trho and len
%
% INPUT:
%         len [1 x 1] or [n x 1] - length scales (per dimension)
%        trho [1 x 1] or [n x 1] - transformed prior variances
%       wwnrm [n x 1] - vector of squared normalized Fourier frequencies
%        
% OUTPUT:
%      cdiag [n x 1] - vector of eigenvalues of C for frequencies in w
%     mdcinv [n x 2] - 1st derivs [dC^-1/dlen, dC^-1/drho]
%        mdc [n x 2] - 1st derivs [dC/dlen, dC/drho]
%    mddcinv [n x 3] - 2nd derivs of C^-1 w.r.t [dlen^2, dlen*drho,drho^2]


% Compute diagonal of ASD covariance matrix
cdiag = trho.*exp(-.5*wwsq.*len.^2);

% 1st derivative of inv(Cdiag)
if nargout > 1
    dcinvdthet = [(len.*wwsq)./cdiag, ...  % dC^-1/dl
        -(1./trho)./cdiag];        % dC^-1/drho
end

% 1st derivative of Cdiag 
if nargout > 2
    dcdthet = [-len.*wwsq.*cdiag, ...   % dC/dl
        (cdiag./trho)]; % dC/drho
        
end

% 2nd derivative of inv(Cdiag)
if nargout > 3
    ddcinvdthet = [(wwsq + len.^2.*wwsq.^2)./cdiag, ... % d^2 C^-1 /dl^2
        -(1./trho).*(len.*wwsq)./cdiag, ...   % d^2 C^-1 /drho dl
        (2./(trho.^2.*cdiag))]; % d^2 C^-1 /drho^2
end
