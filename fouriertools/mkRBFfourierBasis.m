function [Bfft,wwsq] = mkRBFfourierBasis(xx,len,fdprs)
% Compute Fourier-domain basis for RBF covariance on circular interval
%
% [Bfft,wwsq] = mkFourierBasis(x,len,FDprs);
%
% INPUT:
% -----
%            xx [nx x d] - spatial input locations in a d-dimensional space (d<=3)
%           len [1 x d]  - minimum length scale for each dimension
%         fdprs (struct) - struct governing Fourier domain representation
%             .circinterval [2 x d] - circular support in each stimulus dimension
%             .condthresh   [1 x 1] - condition number for thresholding for small eigenvalues 
%
% OUTPUT:
% ------
%       B [nx x nb] - columns define orthogonal basis for C (on Reals)
%    wwsq [nb x 1] - vector of Fourier frequencies
%      ii [nb x 1] - binary vector indicating which frequencies included in U, sdiag

% extract size of input data
[nx,nd] = size(xx);  % number of input dimensions

% flip xx to column vector if necessary
if (nx==1)
    xx = xx';
    nd = 1;
end

% replicate lens to be same length as # of dimensions if necessary
if length(len)==1 && nd>1
    len = len*ones(1,nd);
end

% Compute range for each input dimension
circrnge = diff(fdprs.circinterval); 

% Make Fourier basis for each input dimension 
Bmats = cell(nd,1); % Fourier basis matrix for each filter dimension
wwnrmvecs = cell(nd,1); % Fourier frequencies for each filter dimension
cdiagvecs = cell(nd,1); % Fourier frequencies for each filter dimension

for jj = 1:nd
    % determine maximal freq for Fourier representation
    maxfreq = floor(circrnge(jj)/(pi*len(jj))*sqrt(.5*log(fdprs.condthresh)));

    % Compute basis for non-uniform DFT and frequency vector
    [Bmats{jj},wvecs] = realnufftbasis(xx(:,jj),circrnge(jj),maxfreq*2+1);
    wwnrmvecs{jj} = (2*pi/circrnge(jj))^2*(wvecs.^2); % normalized freqs squared
    cdiagvecs{jj} = exp(-.5*wwnrmvecs{jj}*len(jj).^2); % diagonal of cov

end

% Assemble basis
switch nd   
    case 1  % 1 dimension

        Bfft = Bmats{1}';      % FFT matrix
        wwsq = wwnrmvecs{1}; % normalized freqs squared
        
    case 2   % 2 dimensions
        nfreq = cellfun(@length,wwnrmvecs); % number of frequencies for each dimension
        Cdiag = kron(cdiagvecs{2},cdiagvecs{1}); % diagonal for full space
        ii = Cdiag>1/fdprs.condthresh; % find indices of Fourier coefficients to keep
        [i1,i2] = find(reshape(ii,nfreq')); % find indices for wwnrm1 and wwnrm2
        % Compute outer product of basis vecs to get basis for 2D NUDFT
        Bfft = (Bmats{1}(i1,:).*Bmats{2}(i2,:))'; % assemble basis via Kronecker prod (slowest step)
        wwsq = [wwnrmvecs{1}(i1), wwnrmvecs{2}(i2)];

    case 3   %  3 dimensions
        nfreq = cellfun(@length,wwnrmvecs); % number of frequencies for each dimension
        Cdiag = kron(cdiagvecs{3},(kron(cdiagvecs{2},cdiagvecs{1})));
        ii = Cdiag>1/fdprs.condthresh; % find indices of Fourier coefficients to keep
        [i1,i2and3] = find(reshape(ii,nfreq')); % find indices
        [i2, i3] = ind2sub(nfreq(2:3), i2and3);
        Bfft = (Bmats{1}(i1,:).*Bmats{2}(i2,:).*Bmats{3}(i3,:))'; % assemble basis via Kronecker prod
        wwsq = [wwnrmvecs{1}(i1), wwnrmvecs{2}(i2), wwnrmvecs{3}(i3)];
    otherwise
        error('fdGP only supports data of dimension <= 3');    
end