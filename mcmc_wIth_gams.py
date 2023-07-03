#!/usr/bin/env python
# coding: utf-8

# Import Required Packages
# ========================
import os, sys
import pickle
import time

import casadi

import numpy as np
import matplotlib.pyplot as plt

# MCMC (HMC) sampling routines
sys.path.append(os.path.abspath("mcmc"))
from mcmc_sampling import create_hmc_sampler

# Data Hanlder (.data_handlers.load_site_data)
from data_handlers import load_site_data

from mcmc_with_casadi import log_density_function

# Local Debugging flag; remove when all tested
_DEBUG = False

from gams import GamsWorkspace
if GamsWorkspace.api_major_rel_number<42:  # old API structure
    import gdxcc as gdx
    from gams import *
    import gamstransfer as gt
else:  # new API structure
    import gams.core.gdx as gdx
    from gams.control import *
    import gams.transfer as gt
    
# ## Define a log density function suitable for MCMC sampling
#     Note that the log-density is the logarithm of the target density discarding any normalization factor





def main(
    # Configurations/Settings
    site_num          = 100,  # Number of sites(10, 25, 100, 1000)
    norm_fac          = 1e9,
    delta_t           = 0.02,
    alpha             = 0.045007414,
    kappa             = 2.094215255,
    pf                = 20.76,
    pa                = 44.75,
    xi                = 0.01,
    zeta              = 1.66e-4*1e9,  # zeta := 1.66e-4*norm_fac  #
    #
    max_iter          = 200,
    tol               = 0.01,
    T                 = 200,
    N                 = 200,
    #
    sample_size       = 1000,    # simulations before convergence (to evaluate the mean)
    mode_as_solution  = False,   # If true, use the mode (point of high probability) as solution for gamma
    final_sample_size = 100_00,  # number of samples to collect after convergence
    two_param_uncertainty = False 
    ):
    """
    Main function; putting things together

    :param float tol: convergence tolerance
    :param T:
    :param N:
    """


    # Load sites' data
    (
        zbar_2017,
        gamma,
        gammaSD,
        z_2017,
        forestArea_2017_ha,
        theta,
        thetaSD,
    ) = load_site_data(site_num, norm_fac=norm_fac, )


    # Evaluate Gamma values ()
    gamma_1_vals  = gamma -  gammaSD
    gamma_2_vals  = gamma +  gammaSD
    size    = gamma.size
    # Theta Values
    theta_vals  = theta
    
    



    # Retrieve z data for selected site(s)
    site_z_vals  = z_2017
    
    if two_param_uncertainty == False:
        # Evaluate mean and covariances from site data
        site_stdev       = gammaSD
        site_covariances = np.diag(np.power(site_stdev, 2))
        site_precisions  = np.linalg.inv(site_covariances)
        site_mean        = gamma_1_vals/2 + gamma_2_vals/2

        # Initialize Gamma Values
        uncertain_vals      = gamma.copy()
        uncertain_vals_mean = gamma.copy()
        uncertain_vals_old  = gamma.copy()

    elif two_param_uncertainty == True:
        vals = np.concatenate((theta_vals, gamma_vals))
        # Evaluate mean and covariances from site data
        site_stdev       = np.concatenate((theta_SD, gamma_SD))
        site_covariances = np.diag(np.power(site_stdev, 2))
        site_precisions  = np.linalg.inv(site_covariances)
        site_mean        = vals

        # Initialize Gamma Values
        uncertain_vals      = vals.copy()
        uncertain_vals_mean = vals.copy()
        uncertain_vals_old  = vals.copy()
    
    # Householder to track sampled gamma values
    # uncertain_vals_tracker       = np.empty((uncertain_vals.size, sample_size+1))
    # uncertain_vals_tracker[:, 0] = uncertain_vals.copy()
    uncertain_vals_tracker = [uncertain_vals.copy()]

    # Collected Ensembles over all iterations; dictionary indexed by iteration number
    collected_ensembles = {}

    # Track error over iterations
    error_tracker = []

    # Update this parameter (leng) once figured out where it is coming from
    leng = 200
    arr  = np.cumsum(
             np.triu(
             np.ones((leng, leng))
         ),
         axis=1,
    ).T
    Bdym         = (1-alpha) ** (arr-1)
    Bdym[Bdym>1] = 0.0
    Adym         = np.arange(1, leng+1)
    alpha_p_Adym = np.power(1-alpha, Adym)

    # Results dictionary
    results = dict(
        size=size,
        tol=tol,
        T=T,
        N=N,
        norm_fac=norm_fac,
        delta_t=delta_t,
        alpha=alpha,
        kappa=kappa,
        pf=pf,
        pa=pa,
        xi=xi,
        zeta=zeta,
        sample_size=sample_size,
        final_sample_size=final_sample_size,
        mode_as_solution=mode_as_solution,
    )

    # Initialize error & iteration counter
    error = np.infty
    cntr = 0

    # Loop until convergence
    while cntr < max_iter and error > tol:
        if two_param_uncertainty == False:

            # Update x0
            x0_vals = gamma_vals * forestArea_2017_ha

            x0data = pd.DataFrame(x0_vals)
            x0data.to_csv('X0Data.csv')

            gammadata = pd.DataFrame(uncertain_vals)

            gammadata.to_csv('GammaData.csv')

            cwd = os.getcwd() # get current working directory

            ws = GamsWorkspace(system_directory=r"C:\GAMS\43", working_directory=cwd)
            t1 = ws.add_job_from_file(f"amazon_{size}sites.gms")
            t1.run()
            dfu = pd.read_csv('amazon_data_u.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfu=dfu.drop('T/R ', axis=1)
            sol_val_Up =dfu.to_numpy()

            dfw = pd.read_csv('amazon_data_w.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfw =dfw.drop('T   ', axis=1)
            dfw_np = dfw.to_numpy()

            dfx = pd.read_csv('amazon_data_x.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfx =dfx.drop('T   ', axis=1)
            dfx_np = dfx.to_numpy()

            dfz = pd.read_csv('amazon_data_z.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfz=dfz.drop('T/R ', axis=1)
            dfz_np =dfz.to_numpy()

            sol_val_Ua = dfw_np**2
            sol_val_X = np.concatenate((dfz_np.T, dfx_np.T))
        elif two_param_uncertainty == True:
            # Update x0
            x0_vals = gamma_vals * forestArea_2017_ha

            x0data = pd.DataFrame(x0_vals)
            x0data.to_csv('X0Data.csv')

            gammadata = pd.DataFrame(uncertain_vals[size:])

            gammadata.to_csv('GammaData.csv')
            
            thetadata = pd.DataFrame(uncertain_vals[0:size])
            thetadata.to_csv('ThetaData.csv')

            cwd = os.getcwd() # get current working directory

            ws = GamsWorkspace(system_directory=r"C:\GAMS\43", working_directory=cwd)
            t1 = ws.add_job_from_file(f"amazon_{size}sites_2_param.gms")
            t1.run()
            dfu = pd.read_csv('amazon_data_u.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfu=dfu.drop('T/R ', axis=1)
            sol_val_Up =dfu.to_numpy()

            dfw = pd.read_csv('amazon_data_w.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfw =dfw.drop('T   ', axis=1)
            dfw_np = dfw.to_numpy()

            dfx = pd.read_csv('amazon_data_x.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfx =dfx.drop('T   ', axis=1)
            dfx_np = dfx.to_numpy()

            dfz = pd.read_csv('amazon_data_z.dat', delimiter='\t')
            # Process the data using the pandas DataFrame
            dfz=dfz.drop('T/R ', axis=1)
            dfz_np =dfz.to_numpy()

            sol_val_Ua = dfw_np**2
            sol_val_X = np.concatenate((dfz_np.T, dfx_np.T))

        ## Start Sampling
        # Update signature of log density evaluator
        log_density = lambda uncertain_val: log_density_function(uncertain_val=uncertain_val,
                                                             uncertain_vals_mean=uncertain_vals_mean,
                                                             theta_vals=theta_vals,
                                                             site_precisions=site_precisions,
                                                             alpha=alpha,
                                                             N=N,
                                                             # sol=sol,
                                                             sol_val_X=sol_val_X,
                                                             sol_val_Ua=sol_val_Ua,
                                                             sol_val_Up=sol_val_Up,
                                                             zbar_2017=zbar_2017,
                                                             forestArea_2017_ha=forestArea_2017_ha,
                                                             norm_fac=norm_fac,
                                                             alpha_p_Adym=alpha_p_Adym,
                                                             Bdym=Bdym,
                                                             leng=leng,
                                                             T=T,
                                                             ds_vect=ds_vect,
                                                             zeta=zeta,
                                                             xi=xi,
                                                             kappa=kappa,
                                                             pa=pa,
                                                             pf=pf,
                                                             two_param_uncertainty =  two_param_uncertainty
                                                             )

        # Create MCMC sampler & sample, then calculate diagnostics
        sampler = create_hmc_sampler(
            size=size,
            log_density=log_density,
            #
            burn_in=100,
            mix_in=2,
            symplectic_integrator='verlet',
            symplectic_integrator_stepsize=1e-1,
            symplectic_integrator_num_steps=3,
            mass_matrix=1e+1,
            constraint_test=lambda x: True if np.all(x>=0) else False,
        )
        gamma_post_samples = sampler.sample(
            sample_size=sample_size,
            initial_state=uncertain_vals,
            verbose=True,
        )
        gamma_post_samples = np.asarray(gamma_post_samples)

        # Update ensemble/tracker
        collected_ensembles.update({cntr: gamma_post_samples.copy()})

        # Update gamma value
        weight     = 0.25  # <-- Not sure how this linear combination weighting helps!
        if mode_as_solution:
            raise NotImplementedError("We will consider this in the future; trace sampled points and keep track of objective values to pick one with highest prob. ")

        else:
            uncertain_vals = weight * np.mean(gamma_post_samples, axis=0 ) + (1-weight) * uncertain_vals_old
        uncertain_vals_tracker.append(uncertain_vals.copy())

        # Evaluate error for convergence check
        error = np.max(np.abs(uncertain_vals_old-uncertain_vals) / uncertain_vals_old)
        error_tracker.append(error)
        print(f"Iteration [{cntr+1:4d}]: Error = {error}")

        # Exchange gamma values (for future weighting/update & error evaluation)
        uncertain_vals_old = uncertain_vals

        # Increase the counter
        cntr += 1

        results.update({'cntr': cntr,
                        'error_tracker':np.asarray(error_tracker),
                        'uncertain_vals_tracker': np.asarray(uncertain_vals_tracker),
                        'collected_ensembles':collected_ensembles,
                        })
        pickle.dump(results, open('results.pcl', 'wb'))

        # Extensive plotting for monitoring; not needed really!
        if False:
            plt.plot(uncertain_vals_tracker[-2], label=r'Old $\gamma$')
            plt.plot(uncertain_vals_tracker[-1], label=r'New $\gamma$')
            plt.legend()
            plt.show()

            for j in range(size):
                plt.hist(gamma_post_samples[:, j], bins=50)
                plt.title(f"Iteration {cntr}; Site {j+1}")
                plt.show()

    print("Terminated. Sampling the final distribution")
    # Sample (densly) the final distribution
    final_sample = sampler.sample(
        sample_size=final_sample_size,
        initial_state=uncertain_vals,
        verbose=True,
    )
    final_sample = np.asarray(final_sample)
    results.update({'final_sample': final_sample})
    pickle.dump(results, open('results.pcl', 'wb'))

    return results




