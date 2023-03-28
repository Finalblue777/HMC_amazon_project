#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylab import *
from casadi import *
import time
# Required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Import MCMC sampling routine
import os, sys
sys.path.append(os.path.abspath("mcmc"))
from mcmc_sampling import create_hmc_sampler

# In[ ]:


normalization = 1e9
δ  = 0.02
α  = 0.045007414
κ  = 2.094215255
pf = 20.76
ζ  = 1.66e-4 * normalization
p2 =  44.75
ξ = 15


# In[3]:


#Site Data
df = pd.read_csv("data/calibration_10SitesModel.csv")
z̄ = (df['zbar_2017_10Sites'].to_numpy() )/normalization
probability_space_size = n = len(z̄)

γ1_list  = df['gamma_10Sites'].to_numpy() -  df['gammaSD_10Sites'].to_numpy()
γ2_list  = df['gamma_10Sites'].to_numpy() +  df['gammaSD_10Sites'].to_numpy()
σ_list = df['gammaSD_10Sites'].to_numpy() *np.ones((1,n))
cov_list = σ_list * np.identity(n) *σ_list
mean = γ1_list/2 + γ2_list/2

z0_list = df['z_2017_10Sites'].to_numpy()
γ_post_list = ((γ1_list/2 + γ2_list/2)*np.ones((1,n))).T

θ_list  = df['theta_10Sites'].to_numpy()
Z0_list = z0_list/ normalization


γ_list_mean_posterior = ((γ1_list/2 + γ2_list/2)*np.ones((1,n))).T
γ_post_list_old = γ_post_list


# In[4]:


def normal(z, μ, Σ):
    """
    The density function of multivariate normal distribution.

    Parameters
    ---------------
    z: ndarray(float, dim=2)
        random vector, N by 1
    μ: ndarray(float, dim=1 or 2)
        the mean of z, N by 1
    Σ: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1
    """


    N = 10

    temp2 = -.5 * (z - μ).T @ np.linalg.inv(Σ) @ (z - μ)

    return  temp2


# In[5]:


sample_size = simulation = 1000_00
γ_vec  = np.zeros((10,simulation+1))
γ_vec[:,0]= ((γ1_list/2 + γ2_list/2)).T


# In[6]:


error = 1e9
tol = 0.02


# In[7]:


leng=200
arr = np.cumsum(
                   np.triu(
                     np.ones((leng, leng))
                   ), axis=1)

Bdym=(1-α)**(arr-1)
Bdym[Bdym>1] = 0
Bdym = Bdym.T

Adym = (np.linspace(1,200,200)*np.ones((1,200))).T

# # Explaining the Code
#
#
# 1. Optimization Phase: For a given $\tilde \gamma$ we solve the problem, using some sort of optimization algorithm (IPOPT in our case),
#
# \begin{equation}
#       \left\{ \int_0^\infty \exp(-\delta t) \left[-P^e  \left (\kappa\sum_{i=1}^I Z^i_t- \sum_{i=1}^I \dot X^i_t \right)+  P^a_t  \sum_i \theta^i Z^i_t-\frac \zeta 2 \left (\sum_i U_t^i + V_t^i \right)^2 \right ] dt\right\}
# \end{equation}
#
# \begin{equation} \label{eq:z}
# \dot Z_t^i = U_t^i - V_t^i .
# \end{equation}
#
# \begin{equation} \label{eq:x}
# {\dot X}_t^i  = - \tilde \gamma^i U^i_t - \alpha \left[ X_t^i - \tilde \gamma^i  \left( {{\bar z}^i - Z_t^i }  \right) \right]
# \end{equation}
#
# $$
# X_0^i = \tilde \gamma^i * C
# $$
#
# where $C$ is some constant.
#
# 2. MC phase: The algorithm for this part is the standard Metropolis-Hastings Algorirthm. The only thing that is a bit troublesome to deal with is the formulation of our likelihood which is $g$.
#
#     1. We are given $\gamma^*$ via the MHMC algorithm.
#     2. Use $U_t^i$ and $V_t^i$ to evaluate the objective function under $\gamma^*$.
#     3. Form the Likelihood $g$
#
# \begin{equation}\label{min_solution}
# g^* = \exp\left[ - {\frac 1 \xi } \left\{ \int_0^\infty \exp(-\delta t) \left[-P^e  \left (\kappa\sum_{i=1}^I Z^i_t- \sum_{i=1}^I \dot X^i_t \right)+  P^a_t  \sum_i \theta^i Z^i_t-\frac \zeta 2 \left (\sum_i U_t^i + V_t^i \right)^2 \right ] dt\right\} \right]
# \end{equation}
#
#
#

# In[8]:


era = 0
while error > tol:

    x0_list = γ_post_list.T * df['forestArea_2017_ha_10Sites'].to_numpy()
    X0_list = x0_list/ normalization
    #Construct Matrix A
    Az = np.zeros((n, n+2))
    Ax = np.zeros((1, n+2))

    Ax[0:1,0:n-0] = -α *γ_post_list[0:n].T
    Ax[0, -1] = np.sum(α*γ_post_list.T * z̄)
    Ax[0,-2]  = -α

    A  = np.concatenate((Az, Ax, np.zeros((1, n+2))), axis=0)

    # Construct Matrix B
    Bz = np.identity((n))
    Bx = (np.zeros((1,n)))
    B  = np.concatenate((Bz, Bx,  np.zeros((1, n))), axis=0)

    # Construct Matrix B
    Dz =   np.zeros((n,n))
    Dx = -(np.ones((1,n))*γ_post_list[0:n].T)

    D  = np.concatenate((Dz, Dx, np.zeros((1, n))), axis=0)

    T   = 200
    N   = T

    dt = T/N
    Y = MX.sym('Y'  ,n + 2)
    up = MX.sym('up',n)
    um = MX.sym('um',n)

    rhs = (sparsify(A)@Y + sparsify(B)@(up-um) + sparsify(D)@(up))*dt + Y
    f = Function('f', [Y, um, up],[rhs])

    import math
    ds_vect = np.zeros((N+1,1))
    for i in range(N+1):
        ds_vect[i]=math.exp(-δ*i*dt)

    opti = casadi.Opti()

    # Decision variables for states

    X = opti.variable(n+2 ,N+1)
    # Aliases for states

    Up = opti.variable(n,N)
    Um = opti.variable(n,N)
    Ua = opti.variable(1,N)

    # 1.2: Parameter for initial state
    ic = opti.parameter(n+2-0)

    # Gap-closing shooting constraints
    for k in range(N):
        opti.subject_to(X[:,k+1]==f(X[:,k],Um[:,k], Up[:,k]))

    # Initial and terminal constraints
    opti.subject_to(X[:,0] == ic)
    opti.subject_to(opti.bounded(0,X[0:n,:],z̄[0:n]))
    # Objective: regularization of controls
    # 1.1: added regularization
    for k in range(n-0):
        opti.subject_to(opti.bounded(0,Um[k,:],inf))
        opti.subject_to(opti.bounded(0,Up[k,:],inf))

    opti.subject_to(Ua == sum1(Up+Um)**2 )

    opti.minimize( sum2(ds_vect[0:N,:].T*(Ua* ζ/2 ))
                  - sum2(ds_vect[0:N,:].T*(pf*X[-2,1:] - pf*X[-2,0:-1]  ))
                  - sum2(ds_vect.T*sum1((p2*θ_list - pf*κ )*X[0:n-0,:] )))

    # solve optimization problem
    options = dict()
    options["print_time"] = False
    options["expand"]     = True
    options["ipopt"]      = {
                        'print_level': 0,
                        'fast_step_computation':            'yes',
                        'mu_allow_fast_monotone_decrease':  'yes',
                        'warm_start_init_point':            'yes',
                            }
    opti.solver('ipopt',options)

    t1 = time.time()
    opti.set_value(ic,vertcat(Z0_list,np.sum(X0_list),1))
    sol = opti.solve()


    trace = {"γ":np.zeros((simulation,n))}
    θ_list_comp = θ_list * np.ones((1,n))

    #
    objective_value = -(sum2(ds_vect[0:T,:].T*(sol.value(Ua)* ζ/2 ))
                      - sum2(ds_vect[0:T,:].T*(pf*sol.value(X)[-2,1:]
                                               - pf*sol.value(X)[-2,0:-1]  ))
                      - sum2(ds_vect.T*sum1((p2*θ_list_comp.T - pf*κ )*sol.value(X)[0:n,:] )))

    fγ̄ =   -.5 * (γ_post_list - γ_list_mean_posterior).T @ np.linalg.inv(cov_list) @ (γ_post_list - γ_list_mean_posterior)

    def log_density_value(gamma_val):
        γ_list_prime = np.asarray(gamma_val).flatten()

        # TODO: Handle constraints
        x0_list = γ_list_prime.T * df['forestArea_2017_ha_10Sites'].to_numpy()
        X0_list = x0_list/ normalization

        θ_list_comp = θ_list * np.ones((1,n))
        z̄_comp = z̄ * np.ones((1,n))

        X_dym_list = np.zeros((1,T+1))
        X_zero_list = np.sum(X0_list)*np.ones((200,1))
        X_dym_list[:,0] = np.sum(X0_list)
        ω =   (γ_list_prime.T@(α*z̄_comp.T- α*sol.value(X)[0:n,:-1]) -γ_list_prime.T@sol.value(Up))

        # TODO: FIX the following line, dimensions, and shapes are troublesome!
        # print((Bdym@ω.T ).shape)
        X_dym_list[:,1:] =( (((1-α)**Adym))*X_zero_list  + (Bdym@ω.T )  ).T

        objective_value = -(sum2(ds_vect[0:T,:].T*(sol.value(Ua)* ζ/2 ))
                      - sum2(ds_vect[0:T,:].T*(pf*X_dym_list[:,1:]
                                               - pf*X_dym_list[:,0:-1]  ))
                      - sum2(ds_vect.T*sum1((p2*θ_list_comp.T - pf*κ )*sol.value(X)[0:n,:] )))

        fγ̄ =   -.5 * (γ_list_prime - γ_list_mean_posterior).T @ np.linalg.inv(cov_list) @ (γ_list_prime - γ_list_mean_posterior)
        log_density_val  = -1/ξ * objective_value    + fγ̄

        return log_density_val

    # Create MCMC sampler & sample, then calculate diagnostics
    sampler = create_hmc_sampler(size=probability_space_size,
                                      log_density=log_density_value,
                                      )
    collected_ensemeble = sampler.sample(sample_size=sample_size,
                                        init_state=γ_post_list,
                                        verpose=True,
                                        )
    for i, sample in enumerate(collected_ensemeble):
        trace["γ"][...] = np.asarray(collected_ensemble)

    γ_post_list = (np.sum(trace["γ"], axis=0 )/simulation *np.ones((1,10))).T/4 + γ_post_list_old*3/4
    error =np.max(abs(γ_post_list_old-γ_post_list)/γ_post_list_old)
    print(error)
    γ_post_list_old = γ_post_list
    γ_vec[:, era+1:era+2] = γ_post_list
    Posterior = trace["γ"]
    plt.plot(γ_vec[:,:era+2].T)
    plt.show()

    era = era+1



# In[9]:


np.save(f'ξ_{ξ}_Posterior', Posterior)


# In[10]:


mc = simulation*10


# In[11]:


γ_list_mc = stats.multivariate_normal(mean, cov_list).rvs(size=mc)
γ_list_mc = γ_list_mc[(γ_list_mc >= 0).all(axis=1)]
while np.shape(γ_list_mc)[0] < mc:
    γ_list_mc_temp = stats.multivariate_normal(mean, cov_list).rvs(size = mc - np.shape(γ_list_mc)[0])
    if np.size(γ_list_mc_temp)> 10:
        γ_list_mc_temp = γ_list_mc_temp[(γ_list_mc_temp >= 0).all(axis=1)]
    else:
        γ_list_mc_temp = (γ_list_mc_temp*np.ones((1,10)))
    γ_list_mc = np.concatenate((γ_list_mc, γ_list_mc_temp), axis=0)


# In[12]:


for i in range(10):
    plt.hist(Posterior[:,i], bins = 100, density=True)
    plt.hist(γ_list_mc[:,i], bins = 100, alpha=0.7, density=True)

    plt.title(f'Posterior Distribution Site {i+1}')
    plt.show()


# In[ ]:




