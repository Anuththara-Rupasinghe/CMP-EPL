# Continuous Partitioning of Neuronal Variability

This repository contains the codes for the continuous modulated Poisson (CMP) modeling framework, which partitions neuronal spiking variability over continuous time, by describing a neuron's firing rate as the product of a time-varying stimulus drive and a continuous-time stochastic gain process.

## Forward Model

We assume the following forward model:

The spike counts at the $k^{th}$ time frame and $l^{th}$ trial of the $j^{th}$ stimulus follows an inhomogeneous Poisson process:

$y_{k,j,l} \sim \text{Poisson}\left( \lambda_{k,j,l} dt \right)$

where 

$\lambda_{k,j,l} = \exp\left(x_{k,j} + g_{k,j,l}\right)$

is the underlying neuronal firing rate in Hertz, $dt$ is the chosen bin size in seconds, $g_{k,j,l}$ is the stochastic gain process that models trial-to-trial variability, and $x_{k,j}$ represents the time-varying stimulus drive that is shared among all trials of the $j^{th}$ stimulus presentation. 

Let $\mathbf{x_j} = [x_{1,j}, x_{2,j}, \cdots, x_{K,j}]^\top$ and $\mathbf{g_{j,l}} = [g_{1,j,l}, g_{2,j,l}, \cdots, g_{K,j,l}]^\top$ represent the concatenated time domain latent variables. We then assume that they are distributed as Gaussian Processes via a log-link function:

$\log \mathbf{x_j}  \sim \mathcal{N}\left(\mathbf{0}, \mathbf{S}^{\sf (j)}\right) , S^{\sf (j)}_{k,m} = \rho^{(j)}_S \exp{\left(- \frac{1}{2} \left|{\frac{t_k-t_m}{ \ell^{(j)}_S }}\right|^2\right)}$

where the stimulus drive covariance $\mathbf{S}^{\sf (j)}$ is modeled by the standard RBF kernel and

$\log \mathbf{z_{j,l}} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{G}\right), G_{k,m} = \rho_G \exp{\left(- \frac{1}{2} \left|{\frac{t_k-t_m}{ \ell_G }}\right|^q\right)} $

where the noise covariance $\mathbf{G}$ is modeled by the Exponentiated Power Law kernel and is assumed to be shared across all stimulus conditions. Note that under this generative model, $\theta = [ \rho_G, \ell_G, q, \rho^{(j)}_S, \ell^{(j)}_S, j = 1, \cdots, J ]$ are all hyper-parameters to be inferred.

## Variational Inference

We introduce an efficient variational inference scheme to infer both the hyper-parameters $\theta$, and the latent variables $\mathbf{x_j}$ and $\mathbf{g_{j,l}}$. Further, we reduce the computational complexity of the algorithm by inferring the latent stimulus driven components in Fourier Domain (we formulate a circulant variant of the covariance matrix $\mathbf{S}^{\sf (j)}$ which diagonalizes in the Fourier Domain). Next, we derive the mean and variance of spike counts based on the analytical formulations of our model and compare them with the corresponding empirical statistics of spike count observations. Finally, we compare the performance of our model with several other variants, including a baseline Poisson model, and a variant of the proposed CMP model with the gain covariance also set to be the RBF kernel (i.e., fixing $q = 2$), and two variants of the modulated Poisson model introduces in [Goris et al., 2014].   

## Preliminary requirements for running the code

This code was tested on MATLAB 2018 and should run without any problem on any newer versions.
First, run the setpaths.m file to configure the paths.

## Examples

The folder 'demos' provides simulation examples and real data examples.


 