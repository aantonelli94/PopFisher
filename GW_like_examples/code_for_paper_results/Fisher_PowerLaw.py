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


from scipy.special import erfc
from sympy.parsing import mathematica as M




def Dalpha(Nobs,alpha,Mmin,Mmax):
    
    """
    Estimates without selection effects.
    This expression is only valid for 1D fitting, and crucially it only accounts for the first term in the FM. 
    More accurate estimates can be obtained with the Fisher matrix expression below.
    It is however a useful function to have for its compactness.
    """


    Dalpha_simpified = M.mathematica('Sqrt[1/(Nobs*(alpha^(-2) - (Mmax^alpha*Mmin^alpha*(Log[Mmax] - Log[Mmin])^2)/(Mmax^alpha - Mmin^alpha)^2))]') # from Mathematica notebook.
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




def dpdet_dlambda(theta_samples, Lambda,theta_min,theta_max,NsampsSelFct,var,threshold):     
    """
    TO COMMENT. Returns a number.
    """
    norm = theta_max**Lambda -theta_min**Lambda
    num = norm + norm*Lambda*theta_samples - theta_max**Lambda * Lambda * np.log(theta_max) + theta_min**Lambda* Lambda * np.log(theta_min)
    den = norm * Lambda
    
    arg = pdet_theta(theta_samples,var,threshold) * num/den
    
    return NsampsSelFct**-1*np.sum(arg)  

def ddpdet_ddlambda(theta_samples, Lambda,theta_min,theta_max,NsampsSelFct, var, threshold):  
    """
    TO COMMENT. Returns a number.
    """
    norm = theta_max**Lambda -theta_min**Lambda
    num_1 = - norm**2 + theta_max**Lambda * theta_min**Lambda * Lambda**2 * np.log(theta_max)**2 - 2*theta_max**Lambda * theta_min**Lambda * Lambda**2 * np.log(theta_max)*np.log(theta_min) + theta_max**Lambda * theta_min**Lambda * Lambda**2 * np.log(theta_min)**2
    den_1 = norm**2 * Lambda**2
    num_2 = norm + norm*Lambda*theta_samples - theta_max**Lambda * Lambda * np.log(theta_max) + theta_min**Lambda* Lambda * np.log(theta_min)
    den_2 = norm * Lambda
    
    arg = pdet_theta(theta_samples,var,threshold) * (num_1/den_1 + (num_2/den_2)**2)
    
    return NsampsSelFct**-1*np.sum(arg) 




def FM_1D_powerlaw(theta_samples,Lambda,upperlimit,lowerlimit,var,threshold,NsampsSelFct,Ndetections):
    
    t0 = time.time()
    
    # Load analytical expressions for the arguments of the integrals to solve here.
    
    GammaI_arg   = M.mathematica('alpha^(-2) + (-derselfct^2 + dderselfct*selfct)/selfct^2 - (Mmax^alpha*Mmin^alpha*(Log[Mmax] - Log[Mmin])^2)/(Mmax^alpha - Mmin^alpha)^2')
    GammaII_arg  = M.mathematica('(lnM^(-6 + alpha)*(-(lnM^alpha*((2 + 3*(-2 + alpha)*alpha)*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*(Mmax^alpha*(Log[lnM] - Log[Mmax]) + Mmin^alpha*(-Log[lnM] + Log[Mmin])))^2) - (lnM^3*(Mmax^alpha - Mmin^alpha - (-2 + alpha)*(-1 + alpha)*alpha*lnM^(-3 + alpha)*sigma^2)*(6*(-1 + alpha)*(Mmax^alpha - Mmin^alpha)^2 + Mmax^(2*alpha)*(4 + 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[lnM] - Log[Mmax]))*(Log[lnM] - Log[Mmax]) + Mmin^(2*alpha)*(4 + 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[lnM] - Log[Mmin]))*(Log[lnM] - Log[Mmin]) + Mmax^alpha*Mmin^alpha*(-2*(-2 + alpha)*(-1 + alpha)*alpha*Log[lnM]^2 + 4*(Log[Mmax] + Log[Mmin]) + 2*Log[lnM]*(-4 - 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[Mmax] + Log[Mmin])) + (-2 + alpha)*alpha*((-1 + alpha)*Log[Mmax]^2 + Log[Mmax]*(6 - 4*(-1 + alpha)*Log[Mmin]) + Log[Mmin]*(6 + (-1 + alpha)*Log[Mmin])))))/sigma^2))/(2*(Mmax^alpha - Mmin^alpha)^4*(((-2 + alpha)*(-1 + alpha)*alpha*lnM^(-3 + alpha))/(-Mmax^alpha + Mmin^alpha) + sigma^(-2))^2)')
    GammaIII_arg = M.mathematica('-1/2*(lnM^(-6 + alpha)*(2*lnM^alpha*((2 + 3*(-2 + alpha)*alpha)*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*(Mmax^alpha*(Log[lnM] - Log[Mmax]) + Mmin^alpha*(-Log[lnM] + Log[Mmin])))^2 + (lnM^3*(Mmax^alpha - Mmin^alpha - (-2 + alpha)*(-1 + alpha)*alpha*lnM^(-3 + alpha)*sigma^2)*(6*(-1 + alpha)*(Mmax^alpha - Mmin^alpha)^2 + Mmax^(2*alpha)*(4 + 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[lnM] - Log[Mmax]))*(Log[lnM] - Log[Mmax]) + Mmin^(2*alpha)*(4 + 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[lnM] - Log[Mmin]))*(Log[lnM] - Log[Mmin]) + Mmax^alpha*Mmin^alpha*(-2*(-2 + alpha)*(-1 + alpha)*alpha*Log[lnM]^2 + 4*(Log[Mmax] + Log[Mmin]) + 2*Log[lnM]*(-4 - 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[Mmax] + Log[Mmin])) + (-2 + alpha)*alpha*((-1 + alpha)*Log[Mmax]^2 + Log[Mmax]*(6 - 4*(-1 + alpha)*Log[Mmin]) + Log[Mmin]*(6 + (-1 + alpha)*Log[Mmin])))))/sigma^2))/((Mmax^alpha - Mmin^alpha)^4*(((-2 + alpha)*(-1 + alpha)*alpha*lnM^(-3 + alpha))/(-Mmax^alpha + Mmin^alpha) + sigma^(-2))^3*sigma^2)')
    GammaIV_arg  = M.mathematica('(lnM^(-7 + alpha)*(2*(Mmax^alpha - Mmin^alpha)*(lnM^3*(-Mmax^alpha + Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*lnM^alpha*sigma^2)*((2 + 3*(-2 + alpha)*alpha)*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*(Mmax^alpha*(Log[lnM] - Log[Mmax]) + Mmin^alpha*(-Log[lnM] + Log[Mmin]))) + 2*(1 - alpha)*lnM^alpha*sigma^2*((2 + 3*(-2 + alpha)*alpha)*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*(Mmax^alpha*(Log[lnM] - Log[Mmax]) + Mmin^alpha*(-Log[lnM] + Log[Mmin])))^2 + (1 - alpha)*lnM^3*(Mmax^alpha - Mmin^alpha - (-2 + alpha)*(-1 + alpha)*alpha*lnM^(-3 + alpha)*sigma^2)*(6*(-1 + alpha)*(Mmax^alpha - Mmin^alpha)^2 + Mmax^(2*alpha)*(4 + 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[lnM] - Log[Mmax]))*(Log[lnM] - Log[Mmax]) + Mmin^(2*alpha)*(4 + 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[lnM] - Log[Mmin]))*(Log[lnM] - Log[Mmin]) + Mmax^alpha*Mmin^alpha*(-2*(-2 + alpha)*(-1 + alpha)*alpha*Log[lnM]^2 + 4*(Log[Mmax] + Log[Mmin]) + 2*Log[lnM]*(-4 - 6*(-2 + alpha)*alpha + (-2 + alpha)*(-1 + alpha)*alpha*(Log[Mmax] + Log[Mmin])) + (-2 + alpha)*alpha*((-1 + alpha)*Log[Mmax]^2 + Log[Mmax]*(6 - 4*(-1 + alpha)*Log[Mmin]) + Log[Mmin]*(6 + (-1 + alpha)*Log[Mmin]))))))/(E^((dth - lnM)^2/(2*sigma^2))*(Mmax^alpha - Mmin^alpha)^4*Sqrt[2*Pi]*(((-2 + alpha)*(-1 + alpha)*alpha*lnM^(-3 + alpha))/(-Mmax^alpha + Mmin^alpha) + sigma^(-2))^3*(sigma^2)^(3/2))')
    GammaV_arg   = M.mathematica('-1/2*(2 + (2*(-1 + alpha)^3*lnM^(-6 + alpha)*sigma^2*(7*lnM^3*(Mmax^alpha - Mmin^alpha) + (-1 + alpha)*(4 + (-2 + alpha)*alpha)*lnM^alpha*sigma^2))/(Mmax^alpha - Mmin^alpha)^2 + ((-1 + alpha)^2*lnM^(-6 + alpha)*sigma^2*((-2 + alpha)*(-1 + alpha)*alpha*(Mmax^alpha - Mmin^alpha)*(lnM^3*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*lnM^alpha*sigma^2)*Log[lnM]^2 + (-2 + alpha)*(-1 + alpha)*alpha*Mmax^alpha*(lnM^3*(Mmax^alpha + Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*lnM^alpha*sigma^2)*Log[Mmax]^2 + 2*Mmax^alpha*Log[Mmax]*(-((2 + 5*(-2 + alpha)*alpha)*lnM^3*(Mmax^alpha - Mmin^alpha)) - (-2 + alpha)*(-1 + alpha)*alpha*(2 + (-2 + alpha)*alpha)*lnM^alpha*sigma^2 - 2*(-2 + alpha)*(-1 + alpha)*alpha*lnM^3*Mmin^alpha*Log[Mmin]) + Mmin^alpha*Log[Mmin]*(2*((2 + 5*(-2 + alpha)*alpha)*lnM^3*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*(2 + (-2 + alpha)*alpha)*lnM^alpha*sigma^2) + (-2 + alpha)*(-1 + alpha)*alpha*(lnM^3*(Mmax^alpha + Mmin^alpha) - (-2 + alpha)*(-1 + alpha)*alpha*lnM^alpha*sigma^2)*Log[Mmin]) + 2*Log[lnM]*((Mmax^alpha - Mmin^alpha)*((2 + 5*(-2 + alpha)*alpha)*lnM^3*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*(2 + (-2 + alpha)*alpha)*lnM^alpha*sigma^2) - (-2 + alpha)*(-1 + alpha)*alpha*(lnM^3*(Mmax^alpha - Mmin^alpha) + (-2 + alpha)*(-1 + alpha)*alpha*lnM^alpha*sigma^2)*(Mmax^alpha*Log[Mmax] - Mmin^alpha*Log[Mmin]))))/(Mmax^alpha - Mmin^alpha)^3)/(lnM^2*(((-2 + alpha)*(-1 + alpha)*alpha*lnM^(-3 + alpha))/(-Mmax^alpha + Mmin^alpha) + sigma^(-2))^3*sigma^4)')

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

