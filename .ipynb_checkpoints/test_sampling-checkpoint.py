

# We start putting things together here

import os, sys
sys.path.append(os.path.abspath("mcmc"))

# Import MCMC sampling routine
from mcmc_sampling import create_hmc_sampler


def sample_amazon_posterior(prob_space_size, log_density, sample_size=100, ):
    """
    MCMC sampling fromm the posterior

    :param int prob_space_size: dimension of the probability space to sample
    :param int sample_size: sample size

    :returns: a list with entries being samples from the target posterior
    """
    sampler = create_hmc_sampler(
        size=prob_space_size,
        log_density=log_density,
        # log_density_grad=lambda x: -banana_potential_energy_gradient(x),
    )
    sample = sampler.sample(sample_size=sample_size, initial_state=[0, 0], )

    return sample


