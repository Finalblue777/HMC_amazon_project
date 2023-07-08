#!/usr/bin/env python

# Import Required Packages
# ========================
import os, sys
import pickle
import time

import casadi
import numpy as np
import matplotlib.pyplot as plt

# MCMC (HMC) sampling routines
sys.path.append(os.path.abspath("HMC_amazon_project/mcmc"))
from mcmc_sampling import create_hmc_sampler

import mcmc_with_casadi
# import mcmc_with_gams

# Local Debugging flag; remove when all tested
_DEBUG = False 

########################################################################
import argparse
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--weight",type=float,default=0.25)
parser.add_argument("--xi",type=float,default=0.01)
parser.add_argument("--pf",type=float,default=20.76)
parser.add_argument("--pa",type=float,default=44.75)
parser.add_argument("--theta",type=float,default=1.0)
parser.add_argument("--gamma",type=float,default=1.0)
parser.add_argument("--sitenum",type=int,default=10)
parser.add_argument("--time",type=int,default=200)
parser.add_argument("--dataname",type=str,default="tests")

args = parser.parse_args()
weight = args.weight
pf = args.pf
pa = args.pa
theta_multiplier = args.theta
gamma_multiplier = args.gamma
sitenum = args.sitenum
time = args.time
xi = args.xi
dataname = args.dataname

workdir = os.getcwd()
outputdir = workdir+"/output/"+dataname+"/pf_"+str(pf)+"_pa_"+str(pa)+"_time_"+str(time)+"/theta_"+str(theta_multiplier)+"_gamma_"+str(gamma_multiplier)+"/sitenum_"+str(sitenum)+"_xi_"+str(xi)+"/weight_"+str(weight)+"/"
plotdir = workdir+"/plot/"+dataname+"/pf_"+str(pf)+"_pa_"+str(pa)+"_time_"+str(time)+"/theta_"+str(theta_multiplier)+"_gamma_"+str(gamma_multiplier)+"/sitenum_"+str(sitenum)+"_xi_"+str(xi)+"/weight_"+str(weight)+"/"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

########################################################################

# results = mcmc_with_gams.main(weight = weight,
#                                        xi = xi,
#                                        pf=pf,
#                                        pa=pa,
#                                        site_num=sitenum,
#                                        T=time,
#                                        outputdir=outputdir,
                                    #    )
results = mcmc_with_casadi.main(weight = weight,
                                       xi = xi,
                                       pf=pf,
                                       pa=pa,
                                       site_num=sitenum,
                                       T=time,
                                       outputdir=outputdir,
                                       )
########################################################################
# Plot Error Results
fig, axes = plt.subplots(1, 1, figsize = (8,6))
plt.plot(results['error_tracker'])
plt.xlabel("Iteration")
plt.ylabel("Error")
fig.savefig(plotdir +'error.png', dpi = 100)
plt.close()

# Plot Gamma Estimate Update
fig, axes = plt.subplots(1, 1, figsize = (8,6))
for j in range(results['gamma_size']):
    plt.plot(results['gamma_vals_tracker'][:, j], label=r"$\gamma_{%d}$"%(j+1))
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
fig.savefig(plotdir +'gamma.png', dpi = 100)
plt.close()

# Plot Histograms
fig, axes = plt.subplots(1, 1, figsize = (8,6))
for itr in results['collected_ensembles'].keys():
    for j in range(results['gamma_size']):
        plt.hist(results['collected_ensembles'][itr][:, j], bins=100)
        plt.title(f"Iteration {itr+1}; Site {j+1}")
        fig.savefig(plotdir +'itr_'+str(itr)+'_site_'+str(j)+'.png', dpi = 100)
        plt.close()

# Plot Histogram of the final sample
fig, axes = plt.subplots(1, 1, figsize = (8,6))
for j in range(results['gamma_size']):
    plt.hist(results['final_sample'][:, j], bins=100)
    plt.title(f"Final Sample; Site {j+1}")
    fig.savefig(plotdir +'final_site_'+str(j)+'.png', dpi = 100)
    plt.close()