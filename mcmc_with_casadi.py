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

# Local Debugging flag; remove when all tested
_DEBUG = False

# ## Define a log density function suitable for MCMC sampling
#     Note that the log-density is the logarithm of the target density discarding any normalization factor

# TODO: Daniel will unify the log density interface to enable modeling both gamma (and theta) inference.
def log_density_function(gamma_val,
                         gamma_vals_mean,
                         theta_vals,
                         N,
                         site_precisions,
                         alpha,
                         # sol,
                         sol_val_X,
                         sol_val_Ua,
                         sol_val_Up,
                         zbar_2017,
                         forestArea_2017_ha,
                         norm_fac,
                         alpha_p_Adym,
                         Bdym,
                         leng,
                         T,
                         ds_vect,
                         zeta,
                         xi,
                         kappa,
                         pa,
                         pf,
                         ):
    """
    Define a function to evaluate log-density of the objective/posterior distribution
    Some of the input parameters are updated at each cycle of the outer loop (optimization loop),
    and it becomes then easier/cheaper to udpate the function stamp and keep it separate here
    """

    ds_vect    = np.asarray(ds_vect).flatten()
    gamma_val  = np.asarray(gamma_val).flatten()
    gamma_size = gamma_val.size
    x0_vals    = gamma_val.T.dot(forestArea_2017_ha) / norm_fac
    X_zero     = np.sum(x0_vals) * np.ones(leng)


    # shifted_X = zbar_2017 - sol.value(X)[0:gamma_size, :-1]
    shifted_X  = sol_val_X[0: gamma_size, :-1].copy()
    for j in range(N):
        shifted_X[:, j]  = zbar_2017 - shifted_X[:, j]
    omega      = np.dot(gamma_val, alpha * shifted_X - sol_val_Up)

    X_dym      = np.zeros(T+1)
    X_dym[0]   = np.sum(x0_vals)
    X_dym[1: ] = alpha_p_Adym * X_zero  + np.dot(Bdym, omega.T)

    z_shifted_X = sol_val_X[0: gamma_size, :].copy()
    scl = pa * theta_vals - pf * kappa
    for j in range(N+1):
        z_shifted_X [:, j] *= scl

    term_1 = - np.sum(ds_vect[0: T] * sol_val_Ua) * zeta / 2
    term_2 =   np.sum(ds_vect[0: T] * (X_dym[1: ] - X_dym[0: -1])) * pf
    term_3 =   np.sum(ds_vect * np.sum(z_shifted_X, axis=0))

    obj_val = term_1 + term_2 + term_3

    gamma_val_dev   = gamma_val - gamma_vals_mean
    norm_log_prob   =   - 0.5 * np.dot(gamma_val_dev,
                                       site_precisions.dot(gamma_val_dev)
                                       )
    log_density_val = -1.0  / xi * obj_val + norm_log_prob

    log_density_val = float(log_density_val)

    if _DEBUG:
        print("Term 1: ", term_1)
        print("Term 2: ", term_2)
        print("Term 3: ", term_3)
        print("obj_val: ", obj_val)
        print("norm_log_prob", norm_log_prob)
        print("log_density_val", log_density_val)

    return log_density_val


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
    gamma_size    = gamma.size

    # Evaluate mean and covariances from site data
    site_stdev       = gammaSD
    site_covariances = np.diag(np.power(site_stdev, 2))
    site_precisions  = np.linalg.inv(site_covariances)
    site_mean        = gamma_1_vals/2 + gamma_2_vals/2

    # Retrieve z data for selected site(s)
    site_z_vals  = z_2017

    # Initialize Gamma Values
    gamma_vals      = gamma.copy()
    gamma_vals_mean = gamma.copy()
    gamma_vals_old  = gamma.copy()

    # Theta Values
    theta_vals  = theta

    # Householder to track sampled gamma values
    # gamma_vals_tracker       = np.empty((gamma_vals.size, sample_size+1))
    # gamma_vals_tracker[:, 0] = gamma_vals.copy()
    gamma_vals_tracker = [gamma_vals.copy()]

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

    # Initialize Blocks of the A matrix those won't change
    A  = np.zeros((gamma_size+2, gamma_size+2))
    Ax = np.zeros(gamma_size+2)

    # Construct Matrix B
    B = np.eye(N=gamma_size+2, M=gamma_size, k=0)
    B = casadi.sparsify(B)

    # Construct Matrxi D constant blocks
    D  = np.zeros((gamma_size+2, gamma_size))

    # time step!
    dt = T / N

    # Other placeholders!
    ds_vect = np.exp(- delta_t * np.arange(N+1) * dt)
    ds_vect = np.reshape(ds_vect, (ds_vect.size, 1))

    # Results dictionary
    results = dict(
        gamma_size=gamma_size,
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

        # Update x0
        x0_vals = gamma_vals * forestArea_2017_ha / norm_fac

        # Construct Matrix A from new gamma_vals
        A[: -2, :]        = 0.0
        Ax[0: gamma_size] = - alpha * gamma_vals[0: gamma_size]
        Ax[-1]            = alpha * np.sum(gamma_vals * zbar_2017)
        Ax[-2]            = - alpha
        A[-2, :]          = Ax
        A[-1, :]          = 0.0
        A = casadi.sparsify(A)

        # Construct Matrix D from new gamma_vals
        D[:, :]  = 0.0
        D[-2, :] = -gamma_vals
        D = casadi.sparsify(D)

        # Define the right hand side (symbolic here) as a function of gamma
        gamma = casadi.MX.sym('gamma' , gamma_size+2)
        up    = casadi.MX.sym('up', gamma_size)
        um    = casadi.MX.sym('um', gamma_size)

        rhs = (A @ gamma + B @ (up-um) + D @ up) * dt + gamma
        f = casadi.Function('f', [gamma, um, up], [rhs])


        ## Define an optimizer and initialize it, and set constraints
        opti = casadi.Opti()

        # Decision variables for states
        X = opti.variable(gamma_size+2, N+1)

        # Aliases for states
        Up = opti.variable(gamma_size, N)
        Um = opti.variable(gamma_size, N)
        Ua = opti.variable(1, N)

        # 1.2: Parameter for initial state
        ic = opti.parameter(gamma_size+2)

        # Gap-closing shooting constraints
        for k in range(N):
            opti.subject_to(X[:, k+1] == f(X[:, k], Um[:, k], Up[:, k]))

        # Initial and terminal constraints
        opti.subject_to(X[:, 0] == ic)
        opti.subject_to(opti.bounded(0,
                                     X[0: gamma_size, :],
                                     zbar_2017[0: gamma_size]
                                     )
                        )

        # Objective: regularization of controls
        for k in range(gamma_size):
            opti.subject_to(opti.bounded(0, Um[k,:], casadi.inf))
            opti.subject_to(opti.bounded(0, Up[k,:], casadi.inf))

        opti.subject_to(Ua == casadi.sum1(Up+Um)**2)

        # Set teh optimization problem
        term1 =   casadi.sum2(ds_vect[0: N, :].T * Ua * zeta / 2)
        term2 = - casadi.sum2(ds_vect[0: N, :].T * (pf * (X[-2, 1: ] - X[-2, 0 :-1])))
        term3 = - casadi.sum2(ds_vect.T * casadi.sum1( (pa * theta_vals - pf * kappa ) * X[0: gamma_size, :] ))

        opti.minimize(term1 + term2 + term3)

        # Solve optimization problem
        options               = dict()
        options["print_time"] = True
        options["expand"]     = True
        options["ipopt"]      = {'print_level':                      1,
                                 'fast_step_computation':            'yes',
                                 'mu_allow_fast_monotone_decrease':  'yes',
                                 'warm_start_init_point':            'yes',
                                 }
        opti.solver('ipopt', options)

        opti.set_value(ic,
                       casadi.vertcat(site_z_vals,
                                      np.sum(x0_vals),
                                      1),
                       )

        if _DEBUG:
            print("ic: ", ic)
            print("site_z_vals: ", site_z_vals)
            print("x0_vals: ", x0_vals)
            print("casadi.vertcat(site_z_vals,np.sum(x0_vals),1): ", casadi.vertcat(site_z_vals,np.sum(x0_vals),1))

        # TODO: Discuss with Daniel how this is taking too long, not the sampling!
        print("solving the Outer Optimization problem")
        start_time = time.time()
        sol = opti.solve()
        print(f"Done; time taken {time.time()-start_time} seconds...")

        if _DEBUG:
            print("sol.value(X)", sol.value(X))
            print("sol.value(Ua)", sol.value(Ua))
            print("sol.value(Up)", sol.value(Up))
            print("sol.value(Um)", sol.value(Um))


        # Extract information from the solver
        N          = X.shape[1]-1
        sol_val_X  = sol.value(X)
        sol_val_Up = sol.value(Up)
        sol_val_Ua = sol.value(Ua)

        ## Start Sampling
        # Update signature of log density evaluator
        log_density = lambda gamma_val: log_density_function(gamma_val=gamma_val,
                                                             gamma_vals_mean=gamma_vals_mean,
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
                                                             )

        # Create MCMC sampler & sample, then calculate diagnostics
        sampler = create_hmc_sampler(
            size=gamma_size,
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
            initial_state=gamma_vals,
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
            gamma_vals = weight * np.mean(gamma_post_samples, axis=0 ) + (1-weight) * gamma_vals_old
        gamma_vals_tracker.append(gamma_vals.copy())

        # Evaluate error for convergence check
        error = np.max(np.abs(gamma_vals_old-gamma_vals) / gamma_vals_old)
        error_tracker.append(error)
        print(f"Iteration [{cntr+1:4d}]: Error = {error}")

        # Exchange gamma values (for future weighting/update & error evaluation)
        gamma_vals_old = gamma_vals

        # Increase the counter
        cntr += 1

        results.update({'cntr': cntr,
                        'error_tracker':np.asarray(error_tracker),
                        'gamma_vals_tracker': np.asarray(gamma_vals_tracker),
                        'collected_ensembles':collected_ensembles,
                        })
        pickle.dump(results, open('results.pcl', 'wb'))

        # Extensive plotting for monitoring; not needed really!
        if False:
            plt.plot(gamma_vals_tracker[-2], label=r'Old $\gamma$')
            plt.plot(gamma_vals_tracker[-1], label=r'New $\gamma$')
            plt.legend()
            plt.show()

            for j in range(gamma_size):
                plt.hist(gamma_post_samples[:, j], bins=50)
                plt.title(f"Iteration {cntr}; Site {j+1}")
                plt.show()

    print("Terminated. Sampling the final distribution")
    # Sample (densly) the final distribution
    final_sample = sampler.sample(
        sample_size=final_sample_size,
        initial_state=gamma_vals,
        verbose=True,
    )
    final_sample = np.asarray(final_sample)
    results.update({'final_sample': final_sample})
    pickle.dump(results, open('results.pcl', 'wb'))

    return results


