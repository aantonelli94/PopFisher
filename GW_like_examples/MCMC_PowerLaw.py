"""
Created on Sat Nov 13 12:50 2021

@author: Andrea Antonelli.


This code contains a collection of functions to sample the mass spectral index of a population
of GW events following a simple power law mass distribution.

"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import matplotlib as mpl
import seaborn as sns
import time


from scipy.special import erfc
from sympy.parsing import mathematica as M

def model(lnM,alpha,M_max, M_min):
    
    """
    Model for p(d|lambda).
    
    Input:
    
    - data    (The noisy generated data where we expect the masses to be.)
    - alpha   (Spectral index of the mass distribution.)
    
    """
    
    M = np.exp(lnM) # mass observations
    norm  = alpha/(M_max**alpha-M_min**alpha)
    
    
    return norm*M**(alpha-1)



def log_M(alpha, N, Mmax,Mmin):

    """
    Analytical function to draw samples following a power law.
    
    Inputs:
    
    - alpha: the spectral index chosen.
    - N: the number of events of the true underlying population.
    - Mmax, Mmin: limits of integration, the minimum and maximum mass of underlying pop.
    
    """
    
    return alpha**-1 * np.log((Mmax**(alpha)-Mmin**(alpha))*np.random.uniform(size=N)+ Mmin**(alpha)*np.ones(N))



def selection_function(Lambda,number_samples_for_integration,upper_limit,lower_limit,noise_var,threshold):
    
    """
    This function is redefined in such a way that the hyperparameter appears as an input.
    It is needed inside the function for the log-likelihood.

    Modified by @Jonathan Gair on Feb 9 2022: now the selection function only draws within 10 sigma of the threshold, 
    and then weights the result by the fraciton of the population that satisfies that criterion.
    This is equivalent to setting pdet(theta) identically zero for theta < dth - 10*sigma.
    """


    samp_min = np.exp(threshold - 10.0*noise_var)
    wt = (upper_limit**Lambda - samp_min**Lambda)/(upper_limit**Lambda - lower_limit**Lambda)
    theta_ij = log_M(Lambda,number_samples_for_integration,upper_limit,samp_min)

    return np.sum(0.5*erfc((threshold-theta_ij)/np.sqrt(2)/noise_var))*wt/number_samples_for_integration



def log_likelihood(params, data, hyperprior_min, hyperprior_max, N_samps_selfunction, M_max, M_min, sigma, dth, N_det):
    
    """
    Function for the population log-likelihood.
    
    Inputs:
    
    - params: the parameter to sample through. 
    - data: the data generated offline.
    - hyperprior_min and max: the lower and upper limits of the priors of the parameter.
    - N_samps_selfunction: number of samples for the Monte Carlo integration of the selection function.
    - M_max, M_min: the upper and lower limits for the source parameter.
    - sigma: noise variance assumed in the noise generation (Gaussian) model.
    - dth: threshold for the observability of data.
    - N_det: number of detected events.
    """
    
    # Build up support for posteriors.
    support = (
                (params[0] >= hyperprior_min)&
                (params[0] <= hyperprior_max)
                )
    
    log_likelihood=0

    # Include the selection function.
    pdet = selection_function(params[0],N_samps_selfunction,M_max,M_min,sigma,dth)
    Nsources = N_det
    

    # Write down the population likelihood here.
    for i in np.arange(Nsources):
        
        ppop_ij = model(data[i,:],params[0],M_max, M_min)           # N_samp-long array for population model.
        sum_ppop = np.sum(ppop_ij)                                  # Internal sum of population model over N_samp.
        log_likelihood += np.log(sum_ppop/pdet)                     # Add the samples drawn over N_obs in the loop
                                                                    # and divide by the selection function at each step.
    
    
   
    #Output without infinities and within the specified hyperpriors (only retain output if supported).
    out = np.where(support,log_likelihood,-np.inf)  
    
    
    # Force nans away.
    if np.isfinite(out):
        return out
    else:
        return -np.inf
    
    
def log_likelihood_nosel(params, data, hyperprior_min, hyperprior_max,  M_max, M_min, sigma, N_det):
    
    """
    Added: Jan 27, 2022.
    Function for the population log-likelihood in the case without selection effects.
    
    Inputs:
    
    - params: the parameter to sample through. 
    - data: the data generated offline.
    - hyperprior_min and max: the lower and upper limits of the priors of the parameter.
    - M_max, M_min: the upper and lower limits for the source parameter.
    - sigma: noise variance assumed in the noise generation (Gaussian) model.
    - N_det: number of detected events.
    """
    
    # Build up support for posteriors.
    support = (
                (params[0] >= hyperprior_min)&
                (params[0] <= hyperprior_max)
                )
    
    log_likelihood=0

    # Include the selection function.
    Nsources = N_det
    

    # Write down the population likelihood here.
    for i in np.arange(Nsources):
        
        ppop_ij = model(data[i,:],params[0],M_max, M_min)           # N_samp-long array for population model.
        sum_ppop = np.sum(ppop_ij)                                  # Internal sum of population model over N_samp.
        log_likelihood += np.log(sum_ppop)                     # Add the samples drawn over N_obs in the loop
                                                                    # and divide by the selection function at each step.
    
    
   
    #Output without infinities and within the specified hyperpriors (only retain output if supported).
    out = np.where(support,log_likelihood,-np.inf)  
    
    
    # Force nans away.
    if np.isfinite(out):
        return out
    else:
        return -np.inf

