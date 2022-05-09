# PopFisher: a Fisher matrix for gravitational-wave population inference

by Jonathan Gair, Andrea Antonelli, and Riccardo Barbieri.

The codes in this repository support the results in the paper [at this link](www.google.com).
This paper has been submitted for publication in *MNRAS*.

> We develop a Fisher matrix for population hyperparameters that accounts for source parameter uncertainties and selection effects, and apply it to test cases of increasing complexity. This repository contains an investigation for a toy (Gaussian) model with normally-distributed data and noise, and a more realistic (GW-like) case in which the spectral index of a mass spectrum of black-hole observations is estimated by both Fisher and numerical methods.

![](GW_like_examples/popfish_examples.pdf)

*Caption for the example figure with the main results.*



## Software implementation


The source code used to generate the results and figures for the toy model are in
the `Gaussian_toy_model` folder; the source code for the GW-like case are in the `GW_like_examples` folder.
The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).



## Dependencies

You will have to install the latest versions of `emcee` and `arviz` to run the jupyter notebooks in this repo.

Here is the citation for `emcee`:

```yaml
@article{ForemanMackey:2012ig, 
author = "Foreman-Mackey, Daniel and Hogg, David W. and Lang, Dustin and Goodman, Jonathan", 
title = "{emcee: The MCMC Hammer}", 
eprint = "1202.3665", 
archivePrefix = "arXiv", 
primaryClass = "astro-ph.IM", 
doi = "10.1086/670067", 
journal = "Publ. Astron. Soc. Pac.", 
volume = "125", 
pages = "306--312", 
year = "2013" }
```


## License

All source code is made available under a GNU General Public License (version 3).
