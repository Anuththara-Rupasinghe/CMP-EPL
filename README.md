# Continuously Partitioning Neuronal Variability

This repository contains the codes for inferring the signal and noise latent Gaussian processes underlying multi-stimuli multi-trial spiking observations, using the CMP-EPL (Continuously-partitioned Modulated Poisson - Exponentiated Power Law) model and inference method

## Forward Model

We assume the following forward model:

The spike counts at the $k^{th}$ time frame and $l^{th}$ trial of the $j^{th}$ stimulus follows an inhomogeneous Poisson process:

$y_{k,j,l} \sim \text{Poisson}\left( \lambda_{k,j,l} dt \right)$

where 

$\lambda_{k,j,l} = \exp\left(x_{k,j} + z_{k,j,l}\right)$

is the underlying neuronal firing rate in Hertz, $dt$ is the chosen bin size in seconds, $z_{k,j,l}$ is the noise component that models trial-to-trial variablity, and $x_{k,j}$ represents the stimulus-locked signal component that is shared among all trials of the $j^{th}$ stimulus presentation. 

Let $\mathbf{x_j} = [x_{1,j}, x_{2,j}, \cdots, x_{K,j}]^\top$ and $\mathbf{z_{j,l}} = [z_{1,j,l}, z_{2,j,l}, \cdots, z_{K,j,l}]^\top$ represent the concatanated time domain latent variables. We then assume that they are distributed as Gaussian Processes via a log-link function:

$\log \mathbf{z_{j,l}} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{N}\right), N_{k,m} = \rho_N \exp{\left(- \frac{1}{2} \left|{\frac{t_k-t_m}{ \ell_N }}\right|^q\right)} $

and

$\log \mathbf{x_j}  \sim \mathcal{N}\left(\mathbf{0}, \mathbf{S}^{\sf (j)}\right) , S^{\sf (j)}_{k,m} = \rho^{(j)}_S \exp{\left(- \frac{1}{2} \left|{\frac{t_k-t_m}{ \ell^{(j)}_S }}\right|^2\right)}$

where $\theta = [ \rho_N, \ell_N, q, \rho^{(j)}_S, \ell^{(j)}_S, j = 1, \cdots, J ]$ are hyper-parameters.

## Variational Inference

We introduce an efficient variational inference scheme to infer both the hyper-parameters $\theta$, and the latent variables $\mathbf{x_j}$ and $\mathbf{z_{j,l}}$. Further, we reduce the computational complexity of the algorithm by inferring the latent Signal components in Fourier Domain (we formulate a circulant variant of the covariance matrix $\mathbf{S}^{\sf (j)}$ which diagonalizes in the Fourier Domain). Next, we derive the mean and variance of spike counts based on the analytical formulations of our model and compare them with the corresponding empirical statistics of spike count observations. Finally, we compare the performance of our model with several other variants, including a baseline Poisson model without any noise component, and a variant of the proposed CMP model with the noise covariance also set to be the RBF kernel (i.e., fixing $q = 2$).   

## Preliminary requirements for running the code

This code was tested on MATLAB 2018 and should run without any problem on any newer versions.
First, run the setpaths.m file to configure the paths.

## Examples

The folder 'demos' provides simulation examples and real data examples.


 