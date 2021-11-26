"""
Created on Sat Nov 13 12:50 2021

@author: Andrea Antonelli.

Functions to calculate the population Fisher matrix for a 1-D power-law population model.

"""


import numpy as np
import matplotlib.pyplot as plt
import emcee
import matplotlib as mpl
import seaborn as sns
import time
import pickle


from scipy.special import erfc
from sympy.parsing import mathematica as M


# Load Mathematica expressions calculated offline. See notes on how to calculate Fisher matrix.
with open('Fisher_expressions.pickle', 'rb') as handle:
        Fisher_expressions = pickle.load(handle)


def Dalpha(Nobs,alpha,Mmin,Mmax):
    
    """
    Estimates without selection effects.
    This expression is only valid for 1D fitting, and crucially it only accounts for the first term in the FM. 
    More accurate estimates can be obtained with the Fisher matrix expression below.
    It is however a useful function to have for its compactness.
    """

    with open('Fisher_expressions.pickle', 'rb') as handle:
        b = pickle.load(handle)

    Dalpha_simpified = M.mathematica(Fisher_expressions['Da_no_sel_firstterm']) # from Mathematica notebook.
    out = Dalpha_simpified.subs([('Nobs',Nobs),('alpha',alpha),('Mmax',Mmax),('Mmin',Mmin)]).evalf()
    
    return float(out)

def gaussian(d,mu,Sigma_sq): 
    
    # My redefinition of gaussian to take the sigma^2 as input.
    
    num = np.exp(-0.5*(d-mu)**2/Sigma_sq)
    den = np.sqrt(2*np.pi*Sigma_sq)
    
    return num/den


# Selection functions and derivatives.

def pdet_lambda(theta_samples,N_samps_selfunction,sigma,dth): 
    
    """
    
    Function for the selection function of the hyperparameters.
    
    Inputs:
    
    - theta_samples: an array of sampled data.
    - N_samps_selfunction: number of samples for the Monte Carlo integration.
    - sigma: noise variance.
    - dth: threshold in the presence of selection effects.
    
    """
    return 0.5*N_samps_selfunction**-1*np.sum(erfc((dth-theta_samples)/np.sqrt(2)/sigma))


def pdet_theta(theta_samples,sigma,dth): 
    
    """
    
    Function for the selection function of the source parameters.
    Given the assumption of a gaussian noise model, we can analytically write down 
    the function without resorting to Monte Carlo integration.
    
    Inputs:
    
    - theta_samples: an array of sampled data.
    - sigma: noise variance.
    - dth: threshold in the presence of selection effects.
    
    """
        
    return 0.5* erfc((dth-theta_samples)/np.sqrt(2)/sigma)




def dpdet_dlambda(theta_samples, Lambda,lowerlimit,upperlimit,NsampsSelFct,var,threshold):     
    
    """
    Function for the first derivative of the selection function. Returns a number.
    """
    
    arg_der_plambda = M.mathematica(Fisher_expressions['arg_der_plambda'])
    arg_list=[]
    
    for lnM in theta_samples:
        
        out = arg_der_plambda.subs([('alpha',Lambda),('Mmax',upperlimit),('Mmin',lowerlimit),('lnM',lnM)]).evalf()
        arg_list.append(float(out))
    
    arg = pdet_theta(theta_samples,var,threshold) * arg_list #set up argument of integral.
    
    return NsampsSelFct**-1*np.sum(arg)  



def ddpdet_ddlambda(theta_samples, Lambda,lowerlimit,upperlimit,NsampsSelFct,var,threshold):  
    
    """
    Function for the second derivative of the selection function. Returns a number.
    """
    
    arg_doubleder_plambda = M.mathematica(Fisher_expressions['arg_doubleder_plambda'])


    arg_list=[]
    
    for lnM in theta_samples:
        
        out = arg_doubleder_plambda.subs([('alpha',Lambda),('Mmax',upperlimit),('Mmin',lowerlimit),('lnM',lnM)]).evalf()
        arg_list.append(float(out))
    
    arg = pdet_theta(theta_samples,var,threshold) * arg_list
    
    return NsampsSelFct**-1*np.sum(arg) 




def FM_1D_powerlaw(theta_samples,Lambda,upperlimit,lowerlimit,var,threshold,NsampsSelFct,Ndetections):
    
    t0 = time.time()
    
    # Load analytical expressions for the arguments of the integrals to solve here. See notes for more details.
    
    GammaI_arg   = M.mathematica(Fisher_expressions['A_theta'])
    GammaII_arg  = M.mathematica(Fisher_expressions['B_theta'])
    GammaIII_arg = M.mathematica(Fisher_expressions['C_theta'])
    GammaIV_arg  = M.mathematica(Fisher_expressions['D_theta'])
    GammaV_arg   = M.mathematica(Fisher_expressions['E_theta'])

    substitutions = [('selfct',pdet_lambda(theta_samples,NsampsSelFct,var,threshold)),
                     ('derselfct',dpdet_dlambda(theta_samples,Lambda,lowerlimit,upperlimit, 
                                                NsampsSelFct, var, threshold)),
                     ('dderselfct',ddpdet_ddlambda(theta_samples,Lambda,lowerlimit,upperlimit, 
                                                   NsampsSelFct, var, threshold)),
                     ('alpha',Lambda),
                     ('Mmax',upperlimit),
                     ('Mmin',lowerlimit),
                     ('sigma',var),
                     ('dth',threshold)
                    ]
    
    
    ###################################
    # MC integration of the first term.
    ###################################
    
    
    GammaI_out = GammaI_arg.subs(substitutions).evalf() 
    GammaI_out = GammaI_out * pdet_theta(theta_samples,var,threshold)/pdet_lambda(theta_samples,NsampsSelFct,var,threshold) # set up argument of integral.
    GammaI_int = float(NsampsSelFct**-1*np.sum(GammaI_out)) # perform the MC integration here.
    
    tI = time.time()
    print('------------------------------------------------')
    print('The first term in the FM takes', int(tI-t0), 's to evaluate')
    print('First integral:',  Ndetections *GammaI_int)
    print('------------------------------------------------')
    
    
    ###################################
    # MC integration of the second term.
    ###################################
    
    
    GammaII_list=[]
    GammaII_arg_temp = GammaII_arg.subs(substitutions).evalf() #Help the loop by evaluating common variables.
    
    for lnM in theta_samples:
        
        # The simpy expressions are evaluated into a list for each sample of the masses.
        out = GammaII_arg_temp.subs([('lnM',lnM)]).evalf()
        GammaII_list.append(float(out))
    
    GammaII_arg = GammaII_list*pdet_theta(theta_samples,var,threshold)/pdet_lambda(theta_samples,NsampsSelFct,var,threshold) #set up argument of integral.
    GammaII_int = NsampsSelFct**-1*np.sum(GammaII_arg)# perform the MC integration here.
    
    tII = time.time()
    print('The second term in the FM takes', int(tII-tI), 's to evaluate')
    print('Second integral:', Ndetections *GammaII_int)
    print('------------------------------------------------')
    
    
    ###################################
    # MC integration of the third term.
    ###################################
    
    
    GammaIII_list=[]
    GammaIII_arg_temp = GammaIII_arg.subs(substitutions).evalf() #Help the loop by evaluating common variables.
    
    for lnM in theta_samples:
        
        # The simpy expressions are evaluated into a list for each sample of the masses.
        out = GammaIII_arg_temp.subs([('lnM',lnM)]).evalf()
        GammaIII_list.append(float(out))
    
    GammaIII_arg = GammaIII_list/pdet_lambda(theta_samples,NsampsSelFct,var,threshold)    #set up argument of integral.
    GammaIII_int = NsampsSelFct**-1*np.sum(GammaIII_arg)# perform the MC integration here.
    
    tIII = time.time()
    print('The third term in the FM takes', int(tIII-tII), 's to evaluate')
    print('Third integral:',  Ndetections *GammaIII_int)
    print('------------------------------------------------')
    
    
    ###################################
    # MC integration of the fourth term.
    ###################################
    
    
    GammaIV_list=[]
    GammaIV_arg_temp = GammaIV_arg.subs(substitutions).evalf() #Help the loop by evaluating common variables.
    
    for lnM in theta_samples:
        
        # The simpy expressions are evaluated into a list for each sample of the masses.
        out = GammaIV_arg_temp.subs([('lnM',lnM)]).evalf()
        GammaIV_list.append(float(out))
    
    GammaIV_arg = GammaIV_list/pdet_lambda(theta_samples,NsampsSelFct,var,threshold)    #set up argument of integral.
    GammaIV_int = NsampsSelFct**-1*np.sum(GammaIV_arg)# perform the MC integration here.
    
    tIV = time.time()
    print('The fourth term in the FM takes', int(tIV-tIII), 's to evaluate')
    print('Fourth integral:', Ndetections *GammaIV_int)
    print('------------------------------------------------')
    
    ###################################
    # MC integration of the fifth term.
    ###################################
    
    
    GammaV_list=[]
    GammaV_arg_temp = GammaV_arg.subs(substitutions).evalf() #Help the loop by evaluating common variables.
    
    for lnM in theta_samples:
        
        # The simpy expressions are evaluated into a list for each sample of the masses.
        out = GammaV_arg_temp.subs([('lnM',lnM)]).evalf()
        GammaV_list.append(float(out))
    
    GammaV_arg = GammaV_list*pdet_theta(theta_samples,var,threshold)/pdet_lambda(theta_samples,NsampsSelFct,var,threshold)    #set up argument of integral.
    GammaV_int = NsampsSelFct**-1*np.sum(GammaV_arg)# perform the MC integration here.
    
    tV = time.time()
    print('The fifth term in the FM takes', int(tV-tIV), 's to evaluate')
    print('Fifth integral:',  Ndetections *GammaV_int)
    print('------------------------------------------------')
    
    return Ndetections * (GammaI_int + GammaII_int + GammaIII_int + GammaIV_int + GammaV_int)

