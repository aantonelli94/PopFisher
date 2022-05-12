## The GW-like example


The codes in this folder could be used to reproduce the results in Sec. 4 of the paper.


Fig.3 is produced in `EMRI_mass_function_with_selectioneffects.ipynb`. 
Fig.4 is produced in `Dalpha_vs_N.ipynb`.

Both codes rely on MCMC sampling functions grouped in `MCMC_PowerLaw.py`. The Fisher prediction (in dashed orange) is obtained with functions grouped in `Fisher_PowerLaw.py`. The Fisher predictions are obtained in a semi-analytical fashion, where the integrals are carried out with Monte Carlo methods, while the integrands are simplified analytically. Details as to how practically the Fisher is calculated can be found in `Notes_1D_FM_PowerLaw.ipynb`, while the analytical integrands are simplified in `Fisher_predictions.nb`.


Finally `EMRI_mass_function.ipynb` contains a basic tutorial on MCMC from a power-law population model in the absence of selection effects. Some of the most basic concepts are explained in the markdown here, and used without further explanation in the production codes mentioned above.

