# Copyright © 2023, UChicago Argonne, LLC
# All Rights Reserved

# Math & Science
import numpy as np

try:
    from scipy import sparse
except(ImportError):
    sparse = None

# Other imports
import time
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import pandas as pd
    from pandas.plotting import autocorrelation_plot
except(ImportError):
    pd = autocorrelation_plot = None
try:
    import seaborn as sns
except(ImportError):
    sns = None

from pyoed import utility
from . import (
    Sampler,
    Proposal,
)

## Module-level variables
_DEBUG                 = False
_CONSTRAINT_MAX_TRIALS = 100  # Maximum number of trials (per point) to satisfy constraints (e.g., bounds) for proposl/sampler


class GaussianProposal(Proposal):
    # A dictionary holding default configurations
    _DEF_CONFIGURATIONS = {
        'size':None,
        'random_seed':123,
        'mean':None,
        'variance':1.0,
        'constraint_test':None,
        'proposal_name':'Gaussian',
    }

    def __init__(self, configs=_DEF_CONFIGURATIONS):
        """
        A class implementing a Gaussian proposal for MCMC sampling.
        :param dict configs: a configurations dictionary wchich accepts the following keys:
            - 'size': (int) dimension of the target distribution to sample
            - random_seed: random seed used when the object is initiated to keep track of random samples
              This is useful for reproductivity.
              If `None`, random seed follows `numpy.random.seed` rules
            - mean: mean of the proposal;
                - None (default); the mean is set to the current state passed to the `sample` or `__cal__` method
                - if not None, the mean of the Gaussian proposal is kept fixed; in this case one should
                  choose large variance to cover the space well.
            'constraint_test': a function that returns a boolean value `True` if sample point satisfy
                any constrints, and `False` otherwise; ignored if `None`, is passed.
        """
        configs = utility.aggregate(configs, self._DEF_CONFIGURATIONS)
        if configs['constraint_test'] is None: configs['constraint_test'] = lambda x: True
        super().__init__(configs)

        # Define additional private parameters
        self._update_covariance_matrix(configs['variance'])

        # maintain a proper random state
        random_seed = self._CONFIGURATIONS['random_seed']
        self._RANDOM_STATE = np.random.RandomState(random_seed).get_state()

    def validate_configurations(self, configs, raise_for_invalid=True):
        """
        A method to check the passed configuratios and make sure they
            are conformable with each other, and with current configurations once combined.
        This guarantees that any key-value pair passed in configs can be properly used

        :param dict configs: a dictionary holding key/value configurations
        :param bool raise_for_invalid: if `True` raise :py:class:`TypeError` for invalid configrations type/key

        :returns: True/False flag indicating whether passed coinfigurations dictionary is valid or not

        :raises: see the parameter `raise_for_invalid`
        """
        if not isinstance(configs, dict):
            print(f"the passed configs must be a dictionary; received {type(configs)}!")
            raise TypeError

        # Set initial value of the validity flag
        is_valid = True

        ## Check that all passed configurations keys are acceptable
        # Get a copy of current configurations, and check all passed arguments are valid
        try:
            current_configs = self._CONFIGURATIONS
        except(AttributeError):
            current_configs = self.__class__._DEF_CONFIGURATIONS

        valid_keys = current_configs.keys()
        for key in configs.keys():
            if key not in valid_keys:
                if raise_for_invalid:
                    print(f"Invalid configurations key {key} passed!")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        # Dict containing all configurations (aggregated with current settings)
        aggr_configs = utility.aggregate(configs, current_configs)

        ## Check the target distribution dimensionality
        size = aggr_configs['size']
        if not (utility.isnumber(size) and int(size) == size and size>0):
            if raise_for_invalid:
                print(f"The size key must be a valid positive integer value; "
                      f"received {size} of type {type(size)}")
                raise TypeError
            else:
                is_valid = False
                return is_valid

        ## Check the proposal mean if given
        proposal_mean = aggr_configs['mean']
        if proposal_mean is None:
            is_valid = True
            return is_valid
        else:
            proposal_mean = np.asarray(proposal_mean).flatten()
            if proposal_mean.size != size:
                if raise_for_invalid:
                    print(f"The proposal mean has wrong size; "
                          f"expected {size}; recevived {proposal_mean.size}")
                    raise TypeError
                else:
                    print("")

        # Test constraint function (if provided)
        constraint_test = aggr_configs['constraint_test']
        if callable(constraint_test):
            try:
                assert constraint_test(np.random.rand(size)) in [False, True], "Constraint test must report True/False!"
            except:
                print("The constraint test didn't work as expected!")
                raise
        else:
            assert constraint_test is None, "`constraint_test` must be either a callable of None!"

        ## Variance/covariance
        variance = aggr_configs['variance']

        if utility.isnumber(variance):
            if variance <= 0:
                if raise_for_invalid:
                    print(f"NonPositive variance/covariance value!")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        elif isinstance(variance, np.ndarray):
            if variance.shape != (size, size):
                if raise_for_invalid:
                    print(f"The variance/covariance matrix found has wrong shape"
                          f" > Expected: matrix/array of shape ({size}, {size})"
                          f" > Found matrix of shape: {variance.shape}")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        elif sparse is not None and isinstance(variance, sparse.spmatrix):
            if variance.shape != (size, size):
                print(f"The variance/covariance matrix has wrong shape of {variance.shape}"
                      f" > Expected: matrix/array of shape ({size}, {size})")
                raise TypeError

        else:
            print(f"Invalid type of the variance/covariance matrix {type(variance)}!"
                  f"Expected, scalar, np array or sparse matrix/array")
            raise TypeError

        # Return the validity falg if this point is reached (all clear)
        return is_valid

    def _update_covariance_matrix(self, variance=None,):
        """
        Update the variance/covariance matrix used for proposing new sample points

        :param variance: the variance/covariance matrix to be used/set.
            If None, the one in the configurations dictionary is used.
                Thus, one can update covariances by calling udpate_configurations(variance=new_value).

        This method defines/updates three variables:
            `_COVARIANCE_MATRIX` and `_COVARIANCE_MATRIX_INV`, `_COVARIANCE_MATRIX_SQRT` which should never be updated manually
        """
        size = self._CONFIGURATIONS['size']
        if variance is not None:
            covariance_matrix = variance
        else:
            covariance_matrix = self._CONFIGURATIONS['variance']

        if utility.isnumber(covariance_matrix):
            if covariance_matrix <= 0:
                print(f"NonPositive covariance value!")
                raise ValueError

            if sparse is not None:
                covariance_matrix = sparse.diags(covariance_matrix*np.ones(size), shape=(size, size))
            else:
                covariance_matrix = np.diag(np.ones(size) * covariance_matrix)

        elif isinstance(covariance_matrix, np.ndarray):
            if covariance_matrix.shape != (size, size):
                print(f"The covariance matrix found has wrong shape"
                      f" > Expected: matrix/array of shape ({size}, {size})"
                      f" > Found matrix of shape: {covariance_matrix.shape}")
                raise TypeError

        elif sparse is not None and isinstance(covariance_matrix, sparse.spmatrix):
            if covariance_matrix.shape != (size, size):
                print(f"The covariance matrix has wrong shape of {covariance_matrix.shape}"
                      f" > Expected: matrix/array of shape ({size}, {size})")
                raise TypeError

        else:
            print(f"Invalid type of the covariance matrix {type(covariance_matrix)}!"
                  f"Expected, scalar, np array or sparse matrix/array")
            raise TypeError

        # Create the inverse of the covariance matrix once.
        if sparse is not None:
            self._COVARIANCE_MATRIX     = sparse.csc_array(covariance_matrix)
            self._COVARIANCE_MATRIX_INV = sparse.linalg.inv(self._COVARIANCE_MATRIX)

        else:
            self._COVARIANCE_MATRIX = np.asarray(covariance_matrix)
            self._COVARIANCE_MATRIX_INV = np.linalg.inv(self._COVARIANCE_MATRIX)

        # Covariance matrix square root (lower Cholesky factor for sampling)
        self._COVARIANCE_MATRIX_SQRT = utility.factorize_spsd_matrix(self._COVARIANCE_MATRIX)

    def generate_white_noise(self, size, truncate=True):
        """
        Generate a standard normal random vector of size `size` with values truncated
            at -/+3 if `truncate` is set to `True`

        :returns: a numpy array of size `size` sampled from a standard multivariate normal
            distribution of dimension `size` with mean 0 and covariance matrix equals
            an identity matrix.

        :remarks:
            - this function returns a numpy array of size `size` even if `size` is set to 1
        """
        # Sample (given the current internal random state), then reset the state, and truncate
        np_state = np.random.get_state()
        np.random.set_state(self.random_state)
        randn_vec = np.random.randn(size)
        self.random_state = np.random.get_state()
        np.random.set_state(np_state)

        if truncate:
            randn_vec[randn_vec>3] = 3
            randn_vec[randn_vec<-3] = -3
        return randn_vec

    def covariance_matrix_matvec(self, momentum):
        """
        Multiply the mass matrix (in the configurations) by the passed momentum
        """
        momentum = np.asarray(momentum).flatten()
        if momentum.size != self._CONFIGURATIONS['size']:
            print(f"The passed momentum has invalid size;"
                  f"received {momentum}, expected {self._CONFIGURATIONS['size']}")
            raise TypeError

        return self._COVARIANCE_MATRIX.dot(momentum)

    def covariance_matrix_inv_matvec(self, momentum):
        """
        Multiply the inverse of the mass matrix (in the configurations) by the passed momentum
        """
        momentum = np.asarray(momentum).flatten()
        if momentum.size != self._CONFIGURATIONS['size']:
            print(f"The passed momentum has invalid size;"
                  f"received {momentum}, expected {self._CONFIGURATIONS['size']}")
            raise TypeError

        return self._COVARIANCE_MATRIX_INV.dot(momentum)

    def covariance_matrix_sqrt_matvec(self, momentum):
        """
        Multiply the Square root (Lower Cholesky factor) of the mass matrix (in the configurations) by the passed momentum
        """
        momentum = np.asarray(momentum).flatten()
        if momentum.size != self._CONFIGURATIONS['size']:
            print(f"The passed momentum has invalid size;"
                  f"received {momentum}, expected {self._CONFIGURATIONS['size']}")
            raise TypeError

        return self._COVARIANCE_MATRIX_SQRT.dot(momentum)

    def sample(self, sample_size=1, verbose=False, initial_state=None, ):
        """
        Generate and return a sample of size `sample_size` given the current/initial state.
            If the `initial_state` is not passed, the proposal mean must be set in configurations,
            otherwise a TypeError is raised
            This method returns a list with each entry representing a sample point from the underlying distribution
        """
        size = self._CONFIGURATIONS['size']
        if initial_state is self._CONFIGURATIONS['mean'] is None:
            print("You must either set mean for the proposal, or pass an initial state to propose around")
            raise TypeError
        elif initial_state is not None:
            mean = np.asarray(initial_state).flatten()
        else:
            mean = self._CONFIGURATIONS['meabn']

        if mean.size != size:
            print("The proposal mean has wrong size/shape!")
            print("Expected state/mean of size {size}; received size {mean.size}")
            raise TypeError

        # Retrieve an constraint test (if assigned)
        constraint_test = self._CONFIGURATIONS['constraint_test']
        if constraint_test is None: constraint_test = lambda x: True

        # In case there are domain constraints, these need to be checked after shifting/scaling
        # We give each sample point a max of _CONSTRAINT_MAX_TRIALS
        max_trials       = _CONSTRAINT_MAX_TRIALS if constraint_test is not None else 1
        trials           = 0
        proposed_samples = []
        while len(proposed_samples) < sample_size:
            success = False
            for _ in range(max_trials):
                sample = self.generate_white_noise(size=size)
                sample = self.covariance_matrix_sqrt_matvec(sample) + mean
                if constraint_test(sample):
                    proposed_samples.append(sample)
                    success = True
                    break
            if not success:
                print(f"Maximum number of trials for constraint satisfaction have been exceeded")
                print(f"{max_trials} trials have been made with no success")
                raise ValueError

        return proposed_samples

    def update_configurations(self, **kwargs):
        """
        Take any set of keyword arguments, and lookup each in
            the configurations, and update as nessesary/possible/valid

        :raises:
            - :py:class:`TypeError` is raised if any of the passed keys in `kwargs` is invalid/unrecognized
        """
        # Aggregate the passed settings to the current configurations
        configs = utility.aggregate(kwargs, self.configurations)

        try:
            self.validate_configurations(configs, raise_for_invalid=True)
        except:
            print("Some of the passed arguments are invalid/unacceptable; check the following traceback")
            raise

        # If variance is passed, update covariance matrix
        if 'variance' in kwargs:
            variance = kwargs['variance']
            self._update_covariance_matrix(variance)

        # If random seed is passed update random seed/state
        if 'random_seed' in kwargs:
            self._RANDOM_STATE = np.random.RandomState(kwargs['random_seed']).get_state()


        # TODO: Extra code to be removed
        if False:
            configs = self._CONFIGURATIONS.copy()

            # Check if new size is passed
            if 'size' in kwargs:
                size = kwargs['size']
            else:
                size = configs['size']
            if not (utility.isnumber(size) and int(size) == size and size>0):
                print(f"The size key must be a valid positive integer value; "
                      f"received {size} of type {type(size)}")
                raise TypeError
            configs['size'] = size

            # Check if a new mean is passed
            if mean in 'kwargs':
                mean = kwargs['mean']
            else:
                mean = congis['mean']
            if mean is not None:
                mean = np.asarray(mean).flatten()
                if mean.size != size:
                    print(f"The proposal mean has wrong size; "
                          f"expected {size}; recevived {proposal_mean.size}")
                    raise TypeError
            configs['mean'] = mean

            #
            if 'constraint_test' in kwargs:
                constraint_test = kwargs['constraint_test']
            else:
                constraint_test = configs['constraint_test']
            if callable(constraint_test):
                assert constraint_test(np.random.rand(size)) in [False, True], "Constraint test must report True/False!"
            else:
                assert constraint_test is None, "`constraint_test` must be either a callable of None!"
                constraint_test = lambda x: True
            configs['constraint_test'] = constraint_test

            if 'random_seed' in kwargs:
                random_seed = kwargs['random_seed']
            else:
                random_seed = configs['random_seed']
            configs['random_seed'] = random_seed

            if 'variance' in kwargs:
                variance = kwargs['variance']
            else:
                variance = configs['variance']
            if utility.isnumber(variance):
                if variance <= 0:
                    print(f"NonPositive variance/covariance value!")
                    raise TypeError

            elif isinstance(variance, np.ndarray):
                if variance.shape != (size, size):
                    print(f"The variance/covariance matrix found has wrong shape"
                          f" > Expected: matrix/array of shape ({size}, {size})"
                          f" > Found matrix of shape: {variance.shape}")
                    raise TypeError

            elif sparse is not None and isinstance(variance, sparse.spmatrix):
                print(f"The variance/covariance matrix has wrong shape of {variance.shape}"
                      f" > Expected: matrix/array of shape ({size}, {size})")
                raise TypeError

            else:
                print(f"Invalid type of the variance/covariance matrix {type(variance)}!"
                      f"Expected, scalar, np array or sparse matrix/array")
                raise TypeError
            # All good, update variance in configurations
            configs['variance'] = variance

            # All configurations passed check
            # Update covariance matrix
            self._update_covariance_matrix()

    @property
    def random_state(self):
        """Get a handle of the current internal random state"""
        return self._RANDOM_STATE
    @random_state.setter
    def random_state(self, value):
        """Update the internal random state"""
        try:
            np_state = np.random.get_state()
            np.random.set_state(value)
            self._RANDOM_STATE = value
            np.random.set_state(np_state)
        except:
            print("Invalid random state passed of type '{0}'".format(type(value)))
            raise TypeError


class MCMCSampler(Sampler):
    """
    Basic class for MCMC sampling with a chosen (Default is a Gaussian) proposal
    """
    # A dictionary holding default configurations
    _DEF_CONFIGURATIONS = {
        'size':None,
        'log_density':None,
        'burn_in':500,
        'mix_in':10,
        'proposal':None,
        'constraint_test':None,
        'random_seed':123,
    }

    def __init__(self, configs=_DEF_CONFIGURATIONS):
        """
        Implementation of the MCMC sampling algorithm with a predefined proposal

        :param dict configs: a configurations dictionary wchich accepts the following keys:
            - 'size': (int) dimension of the target distribution to sample
            - 'log_density': (callable) log of the (unscaled) density function to be sampled
            - random_seed: random seed used when the object is initiated to keep track of random samples
              This is useful for reproductivity.
              If `None`, random seed follows `numpy.random.seed` rules
            - 'burn_in': (int) number of sample points to discard before collecting samples
            - 'mix_in': (int) number of generated samples between accepted ones
                (to descrease autocorrelation)
            - 'proposal':
            'constraint_test': a function that returns a boolean value `True` if sample point satisfy
                any constrints, and `False` otherwise; ignored if `None`, is passed.

        :remarks:
            - Validation of the configurations dictionary is taken care of in the super class
            - If a proposal is passed in the configurations, 'constraint_test' should be set to it.
              If a constraint test is passed both here and in the proposal, a ValueError is raised
              to avoid clashing.

            - References:
        """
        configs = utility.aggregate(configs, self._DEF_CONFIGURATIONS)
        super().__init__(configs)

        # Get a handle over the configurations, and update the proposal if needed
        configs     = self._CONFIGURATIONS
        size        = configs['size']
        random_seed = configs['random_seed']
        if configs['proposal'] is None:
            configs['proposal'] = GaussianProposal(dict(size=size,
                                                        random_seed=random_seed,
                                                       )
                                                  )
        elif not isinstance(configs['proposal'], Proposal):
            print(f"Passed proposal must be an instace of class `Proposal` ")
            print(f"Received {configs['proposal']}")

        # Update proposal's constraint test (if passed here and not there)
        if configs['constraint_test'] is not None and configs['proposal']['constraint_test'] is not None:
            print("A clash between constraint test here and in the proposal can happen")
            print("Either pass teh constraint test here or in the proposal configs exclusively")
            raise ValueError
        elif configs['constraint_test'] is not None:
            configs['proposal'].update_configurations(constraint_test=configs['constraint_test'])

        # Get a handle of the proposal & set it's random seed
        self._PROPOSAL = configs['proposal']
        self._PROPOSAL.update_configurations(random_seed=configs['random_seed'])

        # Define log-density
        self._update_log_density()

        # This class does not maintain a random state on its own as the proposal is the one responsible for generating random samples.
        self._RANDOM_STATE = self._PROPOSAL._RANDOM_STATE

    def validate_configurations(self, configs, raise_for_invalid=True):
        """
        A method to check the passed configuratios and make sure they
            are conformable with each other, and with current configurations once combined.
        This guarantees that any key-value pair passed in configs can be properly used

        :param dict configs: a dictionary holding key/value configurations
        :param bool raise_for_invalid: if `True` raise :py:class:`TypeError` for invalid configrations type/key

        :returns: True/False flag indicating whether passed coinfigurations dictionary is valid or not

        :raises: see the parameter `raise_for_invalid`
        """

        if not isinstance(configs, dict):
            print(f"the passed configs must be a dictionary; received {type(configs)}!")
            raise TypeError

        # Set initial value of the validity flag
        is_valid = True

        ## Check that all passed configurations keys are acceptable
        # Get a copy of current configurations, and check all passed arguments are valid
        try:
            current_configs = self._CONFIGURATIONS
        except(AttributeError):
            current_configs = self.__class__._DEF_CONFIGURATIONS

        valid_keys = current_configs.keys()
        for key in configs.keys():
            if key not in valid_keys:
                if raise_for_invalid:
                    print(f"Invalid configurations key {key} passed!")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        # Dict containing all configurations (aggregated with current settings)
        aggr_configs = utility.aggregate(configs, current_configs)

        ## Check the target distribution dimensionality
        size = aggr_configs['size']
        if not (utility.isnumber(size) and int(size) == size and size>0):
            if raise_for_invalid:
                print(f"The size key must be a valid positive integer value; "
                      f"received {size} of type {type(size)}")
                raise TypeError
            else:
                is_valid = False
                return is_valid

        ## Check MCMC sampling settings
        for var in ['burn_in', 'mix_in']:
            val = aggr_configs[var]
            if not (utility.isnumber(val) and val>=0):
                if raise_for_invalid:
                    print(f"The configurations key {var} must be a valid non-negative integer value; "
                          f"received {val} of type {type(val)}")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid
        aggr_configs['mix_in'] = max(aggr_configs['mix_in'], 1)


        ## Check the test constraint function (if provided)
        constraint_test = aggr_configs['constraint_test']
        if callable(constraint_test):
            try:
                assert constraint_test(np.random.rand(size)) in [False, True], "Constraint test must report True/False!"
            except:
                print("The constraint test didn't work as expected!")
                raise
        else:
            assert constraint_test is None, "`constraint_test` must be either a callable of None!"

        ## Test the passed proposal
        proposal = aggr_configs['proposal']
        if proposal is None:
            pass
        else:
            if not isinstance(proposal, Proposal):
                if raise_for_invalid:
                    print(f"The passed proposal is of invalid type; expected instance of :py:class:`Proposal` or None!; "
                          f"received {proposal} of type {type(proposal)}")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        # Return the validity falg if this point is reached (all clear)
        return is_valid

    def update_configurations(self, **kwargs):
        raise NotImplementedError("TODO...")

    def _update_log_density(self, log_density=None, ):
        """
        Update the function that evaluates the logarithm of the (unscaled) target density function
            and the associated gradient (if given) as described by the configurations dictionary.
            This can be halpful to avoid recreating the sampler for various PDFs.

        This method defines/updates two variables:
            `_LOG_DENSITY` which evalute the value of the log-density function of
            the (unscaled) target distribution
        """
        size = self._CONFIGURATIONS['size']

        # Log-Density function
        if log_density is None:
            log_density = self._CONFIGURATIONS['log_density']

        if not callable(log_density):
            print(f"The 'log_density' found in the configurations is not a valid callable/function!")
            raise TypeError
        try:
            test_vec = np.random.randn(size)
            log_density(test_vec)
        except:
            print(f"Failed to evaluate the log-density using a randomly generated vector")
            raise TypeError

        self._LOG_DENSITY = log_density

    def sample(self, sample_size=1, initial_state=None, full_diagnostics=False, verbose=False, ):
        """
        Generate and return a sample of size `sample_size`.
            This method returns a list with each entry representing a sample point from the underlying distribution
        :param int sample_size:
        :param initial_state:
        :param bool full_diagnostics: if `True` all generated states will be tracked and kept for full disgnostics, otherwise,
            only collected samples are kept in memory
        :param bool verbose:
        """
        mcmc_results = self.start_MCMC_sampling(sample_size=sample_size,
                                                initial_state=initial_state,
                                                full_diagnostics=full_diagnostics,
                                                verbose=verbose,
                                                )
        return mcmc_results['collected_ensemble']

    def start_MCMC_sampling(self,
                            sample_size,
                            initial_state=None,
                            full_diagnostics=False,
                            verbose=False,
                            ):
        """
        Start the HMC sampling procedure with initial state as passed.
        Use the underlying configurations for configuring the Hamiltonian trajectory, burn-in and mixin settings.

        :param int sample_size: number of smaple points to generate/collect from the predefined target distribution
        :param initial_state: initial point of the chain (any point that falls in the target distribution or near by it
            will result in faster convergence). You can try prior mean if this is used in a Bayesian approach
        :param bool randomize_step_size: if `True` a tiny random number is added to the passed step
            size to help improve space exploration
        :param bool full_diagnostics: if `True` all generated states will be tracked and kept for full disgnostics, otherwise,
            only collected samples are kept in memory
        :param bool verbose: screen verbosity
        """

        # Extract configurations from the configurations dictionary
        state_space_dimension = self._CONFIGURATIONS['size']
        burn_in_steps         = self._CONFIGURATIONS['burn_in']
        mixing_steps          = self._CONFIGURATIONS['mix_in']
        constraint_test       = self._CONFIGURATIONS['constraint_test']

        liner, sliner = '=' * 53, '-' * 40
        if verbose:
            print("\n%s\nStarted Sampling\n%s\n" % (liner, liner))

        # Chain initial state
        if initial_state is None:
            initial_state = self.generate_white_noise(state_space_dimension)
        else:
            initial_state = np.array(initial_state).flatten()
            if initial_state.size != state_space_dimension:
                print(f"Passed initial stae has invalid shape/size"
                      f"Passed initial state has size {initial_state.size}"
                      f"Expected size: {state_space_dimension}")
                raise TypeError

        # Setup and construct the chain using HMC proposal:
        chain_length = burn_in_steps + sample_size * mixing_steps

        # Initialize the chain
        current_state = initial_state.copy()  # initial state = ensemble mean

        # All generated sample points will be kept for testing and efficiency analysis
        chain_state_repository   = [initial_state]
        proposals_repository     = []
        acceptance_flags         = []
        acceptance_probabilities = []
        uniform_random_numbers   = []
        collected_ensemble       = []

        # Build the Markov chain
        start_time = time.time()  # start timing
        for chain_ind in range(chain_length):

            ## Proposal step :propose state

            # Advance the current state
            proposed_state = self.proposal(initial_state=current_state, )

            ## MH step (Accept/Reject) proposed state
            # Calculate acceptance proabability
            current_log_prob  = self.log_density(current_state)
            constraint_violated = False
            if constraint_test is not None:
                if not constraint_test(proposed_state): constraint_violated = True

            if constraint_violated:
                acceptance_probability = 0

            else:
                proposal_log_prob = self.log_density(proposed_state)
                energy_loss = current_log_prob - proposal_log_prob
                _loss_thresh = 1000
                if abs(energy_loss) >= _loss_thresh:  # this should avoid overflow errors
                    if energy_loss < 0:
                        sign = -1
                    else:
                        sign = 1
                    energy_loss = sign * _loss_thresh
                acceptance_probability = np.exp(-energy_loss)
                acceptance_probability = min(acceptance_probability, 1.0)

            # a uniform random number between 0 and 1
            np_state = np.random.get_state()
            np.random.set_state(self.random_state)
            uniform_probability = np.random.rand()
            self.random_state = np.random.get_state()
            np.random.set_state(np_state)

            # MH-rule
            if acceptance_probability > uniform_probability:
                current_state   = proposed_state
                accept_proposal = True
            else:
                accept_proposal = False

            if verbose:
                print(f"\rHMC Iteration [{chain_ind+1:4d}/{chain_length:4d}]; Accept Prob: {acceptance_probability:3.2f}; --> Accepted? {accept_proposal}", end="  ")

            #
            if chain_ind >= burn_in_steps and chain_ind % mixing_steps==0:
                collected_ensemble.append(current_state.copy())

            # Update Results Repositories:
            if full_diagnostics:
                proposals_repository.append(proposed_state)
                acceptance_probabilities.append(acceptance_probability)
                uniform_random_numbers.append(uniform_probability)
                #
                if accept_proposal:
                    acceptance_flags.append(1)
                else:
                    acceptance_flags.append(0)
                chain_state_repository.append(np.squeeze(current_state))

        # Stop timing
        chain_time = time.time() - start_time

        # ------------------------------------------------------------------------------------------------

        # Now output diagnostics and show some plots :)
        if full_diagnostics:
            chain_diagnostics = self.mcmc_chain_diagnostic_statistics(
                proposals_repository=proposals_repository,
                chain_state_repository=chain_state_repository,
                collected_ensemble=collected_ensemble,
                acceptance_probabilities=acceptance_probabilities,
                uniform_probabilities=uniform_random_numbers,
                acceptance_flags=acceptance_flags,
            )

        #
        # ======================================================================================================== #
        #                Output sampling diagnostics and plot the results for 1 and 2 dimensions                   #
        # ======================================================================================================== #
        #
        if verbose:
            print("MCMC sampler:")
            print(f"Time Elapsed for MCMC sampling: {chain_time} seconds")
            print(f"Acceptance Rate: {chain_diagnostics['acceptance_rate']:.2f}")

        sampling_results = dict(
            chain_state_repository=chain_state_repository,
            collected_ensemble=collected_ensemble,
            proposals_repository=proposals_repository,
            acceptance_flags=acceptance_flags,
            acceptance_probabilities=acceptance_probabilities,
            uniform_random_numbers=uniform_random_numbers,
            chain_diagnostics=chain_diagnostics,
            chain_time=chain_time,
        )
        return sampling_results

    def log_density(self, state):
        """
        Evaluate the value of the logarithm of the target unscaled posterior density function
        """
        val = self._LOG_DENSITY(state)
        try:
            val[0]
            val = np.asarray(val).flatten()[0]
        except:
            pass
        # if isinstance(val, np.ndarray) and val.size == 1: val = val.flatten()[0]
        return val


    def mcmc_chain_diagnostic_statistics(self,
                                         proposals_repository,
                                         chain_state_repository,
                                         uniform_probabilities,
                                         acceptance_probabilities,
                                         collected_ensemble,
                                         acceptance_flags=None,
                                         ):
        """
        Return diagnostic statistics of the chain such as the rejection rate, acceptance ratio, etc.
        """
        if acceptance_flags is None:
            acceptance_flags = np.asarray(acceptance_probabilities >= uniform_probabilities, dtype=np.int)
        else:
            acceptance_flags = np.asarray(acceptance_flags)
        acceptance_rate = float(acceptance_flags.sum()) / np.size(acceptance_flags) * 100.0
        rejection_rate = (100.0 - acceptance_rate)

        # Plots & autocorrelation, etc.
        plot_mcmc_results_nd(
            collected_ensemble,
            log_density=self.log_density,
            title="MCMC",
            filename_prefix="MCMC_Sampling",
        )
        # TODO: Add More; e.g., effective sample size, etc.

        # Return all diagonistics in a dictionary
        chain_diagnositics = dict(
            acceptance_rate=acceptance_rate,
            rejection_rate=rejection_rate,
        )

        return chain_diagnositics

    @property
    def proposal(self):
        """Get a handle of the proposal"""
        return self._PROPOSAL

    @property
    def random_state(self):
        """Get a handle of the current internal random state"""
        return self._RANDOM_STATE
    @random_state.setter
    def random_state(self, value):
        """Update the internal random state"""
        self._PROPOSAL.random_state = value
        self._RANDOM_STATE = self._PROPOSAL.random_state


class HMCSampler(Sampler):
    # A dictionary holding default configurations
    _DEF_CONFIGURATIONS = {
        'size':None,
        'log_density':None,
        'log_density_grad':None,
        'random_seed':123,
        'burn_in':500,
        'mix_in':10,
        'symplectic_integrator':'verlet',
        'symplectic_integrator_stepsize':1e-2,
        'symplectic_integrator_num_steps':20,
        'mass_matrix':1,
        'constraint_test':None,
    }

    def __init__(self, configs=_DEF_CONFIGURATIONS):
        """
        Implementation of the HMC sampling algorithm
            (with multiple choices of the symplectic integrator).

        :param dict configs: a configurations dictionary wchich accepts the following keys:
            - 'size': (int) dimension of the target distribution to sample
            - 'log_density': (callable) log of the (unscaled) density function to be sampled
            - 'log_density_grad': (callable) the gradient of the `log_density` function passed.
                If None, an attempt will be made to used automatic differentiation
                (if available), otherwise finite differences (FD) will be utilized
            - random_seed: random seed used when the object is initiated to keep track of random samples
              This is useful for reproductivity.
              If `None`, random seed follows `numpy.random.seed` rules
            - 'burn_in': (int) number of sample points to discard before collecting samples
            - 'mix_in': (int) number of generated samples between accepted ones
                (to descrease autocorrelation)
            - 'symplectic_integrator': (str) name of the symplectic integrator to use;
                acceptable are:
                  + 'leapfrog', '2-stage', '3-stage',
                      where both 'leapfrog' and 'verlet' are equivalent
            'symplectic_integrator_stepsize': (positive scalar) the step size of the symplectic
                integrator
            'symplectic_integrator_num_steps': (postive integer) number of steps of size
                `symplectic_integrator_stesize` taken before returnig the next proposed point
                over the Hamiltonian trajectory
            'mass_matrix': (nonzero scalar or SPD array of size `size x size`)  mass matrix
                to be used to adjust sampling the auxilliary Gaussian momentum
            'constraint_test': a function that returns a boolean value `True` if sample point satisfy
                any constrints, and `False` otherwise; ignored if `None`, is passed.

        :remarks:
            - Validation of the configurations dictionary is taken care of in the super class
            - References:
        """
        configs = utility.aggregate(configs, self._DEF_CONFIGURATIONS)
        super().__init__(configs)

        # Define additional private parameters
        self._update_mass_matrix()

        # Define log-density and associated gradient
        self._update_log_density()

        # maintain a proper random state
        random_seed = self._CONFIGURATIONS['random_seed']
        self._RANDOM_STATE = np.random.RandomState(random_seed).get_state()

    def validate_configurations(self, configs, raise_for_invalid=True):
        """
        A method to check the passed configuratios and make sure they
            are conformable with each other, and with current configurations once combined.
        This guarantees that any key-value pair passed in configs can be properly used

        :param dict configs: a dictionary holding key/value configurations
        :param bool raise_for_invalid: if `True` raise :py:class:`TypeError` for invalid configrations type/key

        :returns: True/False flag indicating whether passed coinfigurations dictionary is valid or not

        :raises: see the parameter `raise_for_invalid`
        """

        if not isinstance(configs, dict):
            print(f"the passed configs must be a dictionary; received {type(configs)}!")
            raise TypeError

        # Set initial value of the validity flag
        is_valid = True

        ## Check that all passed configurations keys are acceptable
        # Get a copy of current configurations, and check all passed arguments are valid
        try:
            current_configs = self._CONFIGURATIONS
        except(AttributeError):
            current_configs = self.__class__._DEF_CONFIGURATIONS

        valid_keys = current_configs.keys()
        for key in configs.keys():
            if key not in valid_keys:
                if raise_for_invalid:
                    print(f"Invalid configurations key {key} passed!")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        # Dict containing all configurations (aggregated with current settings)
        aggr_configs = utility.aggregate(configs, current_configs)

        ## Check the target distribution dimensionality
        size = aggr_configs['size']
        if not (utility.isnumber(size) and int(size) == size and size>0):
            if raise_for_invalid:
                print(f"The size key must be a valid positive integer value; "
                      f"received {size} of type {type(size)}")
                raise TypeError
            else:
                is_valid = False
                return is_valid

        ## Check MCMC sampling settings
        for var in ['burn_in', 'mix_in']:
            val = aggr_configs[var]
            if not (utility.isnumber(val) and val>0):
                if raise_for_invalid:
                    print(f"The configurations key {var} must be a valid non-negative integer value; "
                          f"received {val} of type {type(val)}")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid
        aggr_configs['mix_in'] = max(aggr_configs['mix_in'], 1)

        ## Check the log_density function
        log_density = aggr_configs['log_density']
        if not callable(log_density):
            if raise_for_invalid:
                print(f"The 'log_density' is not a valid callable/function")
                raise TypeError
            else:
                is_valid = False
                return is_valid

        # test ability to properly call the log_density function
        try:
            test_vec = np.random.randn(size)
            log_density(test_vec)
        except:
            if raise_for_invalid:
                print(f"Failed to evaluate the log-density using a randomly generated vector")
                raise TypeError
            else:
                is_valid = False
                return is_valid

        # Test constraint function (if provided)
        constraint_test = aggr_configs['constraint_test']
        if callable(constraint_test):
            try:
                assert constraint_test(np.random.rand(size)) in [False, True], "Constraint test must report True/False!"
            except:
                print("The constraint test didn't work as expected!")
                raise
        else:
            assert constraint_test is None, "`constraint_test` must be either a callable of None!"

        ## Mass matrix (covariance of the momentum)
        mass_matrix = aggr_configs['mass_matrix']

        if utility.isnumber(mass_matrix):
            if mass_matrix <= 0:
                if raise_for_invalid:
                    print(f"NonPositive momentum covariance (mass) value!")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        elif isinstance(mass_matrix, np.ndarray):
            if mass_matrix.shape != (size, size):
                if raise_for_invalid:
                    print(f"The mass matrix found has wrong shape"
                          f" > Expected: matrix/array of shape ({size}, {size})"
                          f" > Found matrix of shape: {mass_matrix.shape}")
                    raise TypeError
                else:
                    is_valid = False
                    return is_valid

        elif sparse is not None and isinstance(mass_matrix, sparse.spmatrix):
            if mass_matrix.shape != (size, size):
                print(f"The mass matrix has wrong shape of {mass_matrix.shape}"
                      f" > Expected: matrix/array of shape ({size}, {size})")
                raise TypeError

        else:
            print(f"Invalid type of the mass matrix {type(mass_matrix)}!"
                  f"Expected, scalar, np array or sparse matrix/array")
            raise TypeError

        # TODO: Proceed here
        ## Test the Symplectic symplectic_integrator parameters (symplectic_integrator name, step size, number of steps)
        pass  # TODO:

        # Return the validity falg if this point is reached (all clear)
        return is_valid

    def sample(self, sample_size=1, initial_state=None, full_diagnostics=False, verbose=False, ):
        """
        Generate and return a sample of size `sample_size`.
            This method returns a list with each entry representing a sample point from the underlying distribution
        :param int sample_size:
        :param initial_state:
        :param bool full_diagnostics: if `True` all generated states will be tracked and kept for full disgnostics, otherwise,
            only collected samples are kept in memory
        :param bool verbose:
        """
        hmc_results = self.start_MCMC_sampling(sample_size=sample_size,
                                               initial_state=initial_state,
                                               full_diagnostics=full_diagnostics,
                                               verbose=verbose,
                                               )
        return hmc_results['collected_ensemble']

    def map_estimate(self, sample_size=100, initial_state=None, verbose=False, ):
        """
        Search for a MAP (maximum aposteriori) estimate by sampling (space exploration)
            This method returns a single-point estimate of the MAP of the distribution
        :param int sample_size:
        :param initial_state:
        :param bool verbose:
        """
        hmc_results = self.start_MCMC_sampling(sample_size=sample_size,
                                               initial_state=initial_state,
                                               full_diagnostics=full_diagnostics,
                                               verbose=verbose,
                                               )
        return hmc_results['map_estimate']

    def generate_white_noise(self, size, truncate=True):
        """
        Generate a standard normal random vector of size `size` with values truncated
            at -/+3 if `truncate` is set to `True`

        :returns: a numpy array of size `size` sampled from a standard multivariate normal
            distribution of dimension `size` with mean 0 and covariance matrix equals
            an identity matrix.

        :remarks:
            - this function returns a numpy array of size `size` even if `size` is set to 1
        """
        # Sample (given the current internal random state), then reset the state, and truncate
        np_state = np.random.get_state()
        np.random.set_state(self.random_state)
        randn_vec = np.random.randn(size)
        self.random_state = np.random.get_state()
        np.random.set_state(np_state)

        if truncate:
            randn_vec[randn_vec>3] = 3
            randn_vec[randn_vec<-3] = -3
        return randn_vec

    def mass_matrix_matvec(self, momentum):
        """
        Multiply the mass matrix (in the configurations) by the passed momentum
        """
        momentum = np.asarray(momentum).flatten()
        if momentum.size != self._CONFIGURATIONS['size']:
            print(f"The passed momentum has invalid size;"
                  f"received {momentum}, expected {self._CONFIGURATIONS['size']}")
            raise TypeError

        return self._MASS_MATRIX.dot(momentum)

    def mass_matrix_inv_matvec(self, momentum):
        """
        Multiply the inverse of the mass matrix (in the configurations) by the passed momentum
        """
        momentum = np.asarray(momentum).flatten()
        if momentum.size != self._CONFIGURATIONS['size']:
            print(f"The passed momentum has invalid size;"
                  f"received {momentum}, expected {self._CONFIGURATIONS['size']}")
            raise TypeError

        return self._MASS_MATRIX_INV.dot(momentum)

    def mass_matrix_sqrt_matvec(self, momentum):
        """
        Multiply the Square root (Lower Cholesky factor) of the mass matrix (in the configurations) by the passed momentum
        """
        momentum = np.asarray(momentum).flatten()
        if momentum.size != self._CONFIGURATIONS['size']:
            print(f"The passed momentum has invalid size;"
                  f"received {momentum}, expected {self._CONFIGURATIONS['size']}")
            raise TypeError

        return self._MASS_MATRIX_SQRT.dot(momentum)

    def _update_mass_matrix(self):
        """
        Update the momentum covariance, i.e., the mass matrix given the current mass matrix
            in the configurations dictionary, and the iverse of the mass matrix

        This method defines/updates three variables:
            `_MASS_MATRIX` and `_MASS_MATRIX_INV`, `_MASS_MATRIX_SQRT`  which should never be updated manually
        """
        size = self._CONFIGURATIONS['size']
        mass_matrix = self._CONFIGURATIONS['mass_matrix']

        if utility.isnumber(mass_matrix):
            if mass_matrix <= 0:
                print(f"NonPositive momentum covariance (mass) value!")
                raise ValueError

            if sparse is not None:
                mass_matrix = sparse.diags(mass_matrix*np.ones(size), shape=(size, size))
            else:
                mass_matrix = np.diag(mass_matrix*np.ones(size))

        elif isinstance(mass_matrix, np.ndarray):
            if mass_matrix.shape != (size, size):
                print(f"The mass matrix found has wrong shape"
                      f" > Expected: matrix/array of shape ({size}, {size})"
                      f" > Found matrix of shape: {mass_matrix.shape}")
                raise TypeError

        elif sparse is not None and isinstance(mass_matrix, sparse.spmatrix):
            if mass_matrix.shape != (size, size):
                print(f"The mass matrix has wrong shape of {mass_matrix.shape}"
                      f" > Expected: matrix/array of shape ({size}, {size})")
                raise TypeError

        else:
            print(f"Invalid type of the mass matrix {type(mass_matrix)}!"
                  f"Expected, scalar, np array or sparse matrix/array")
            raise TypeError

        # Create the inverse of the mass matrix once.
        if sparse is not None:
            self._MASS_MATRIX     = sparse.csc_array(mass_matrix)
            self._MASS_MATRIX_INV = sparse.linalg.inv(self._MASS_MATRIX)

        else:
            self._MASS_MATRIX = np.asarray(mass_matrix)
            self._MASS_MATRIX_INV = np.linalg.inv(self._MASS_MATRIX)

        # Mass matrix square root (lower Cholesky factor for sampling)
        self._MASS_MATRIX_SQRT = utility.factorize_spsd_matrix(self._MASS_MATRIX)

    def log_density(self, state):
        """
        Evaluate the value of the logarithm of the target unscaled posterior density function
        """
        val = self._LOG_DENSITY(state)
        try:
            val[0]
            val = np.asarray(val).flatten()[0]
        except:
            pass
        # if isinstance(val, np.ndarray) and val.size == 1: val = val.flatten()[0]
        return val

    def log_density_grad(self, state):
        """
        Evaluate the gradient of the logarithm of the target unscaled posterior density function
        """
        return self._LOG_DENSITY_GRAD(state)

    def _create_func_grad(self, func, size, approach='fd', fd_eps=1e-5, fd_central=False):
        """
        Given a callable/function `func`, create a function that evaluates the gradient of this function
        :param int size: the domain size which determines the size of the returned gradient
        :param str approach: the approach to use for creating the function.

        :remarks: this method is planned to enable automatic differentiation (AD) if available on
            the current platform, otherwise finite differences 'fd' is used
        """
        if re.match(r"\A(f(-|_| )*d|finite(-|_| )*difference(s)*)\Z", approach, re.IGNORECASE):
            def func_grad(x, fd_eps=fd_eps, fd_central=fd_central):
                """Function to generate gradient using finite differences"""
                x    = np.asarray(x).flatten()
                grad = np.zeros_like(x)
                e    = np.zeros_like(x)
                for i in range(e.size):
                    e[:] = 0.0
                    e[i] = fd_eps

                    if fd_central:
                        grad[i] = (func(x+e) - func(x-e)) / (2.0 * fd_eps)
                    else:
                        grad[i] = (func(x+e) - func(x)) / fd_eps
                return grad
            return func_grad

        elif re.match(r"\A(a(-|_| )*d|automatic(-|_| )*differentiation)\Z", approach, re.IGNORECASE):
            raise NotImplementedError("TODO: A/D is not yet supported for creating function gradient")

        else:
            print(f"Unrecognized gradient generation approach {approach}")
            raise ValueError

    def _update_log_density(self):
        """
        Update the function that evaluates the logarithm of the (unscaled) target density function
            and the associated gradient (if given) as described by the configurations dictionary.
            This can be halpful to avoid recreating the sampler for various PDFs.
            If the gradient is not given, either automatic differentiation (if installed/requested)
            is utilized, otherwise finite-differences are used

        This method defines/updates two variables:
            `_LOG_DENSITY` and `_LOG_DENSITY_GRAD` which evalute the value and the gradient of
                the log-density function of the (unscaled) target distribution
        """
        size = self._CONFIGURATIONS['size']

        # Log-Density function
        log_density = self._CONFIGURATIONS['log_density']
        if not callable(log_density):
            print(f"The 'log_density' found in the configurations is not a valid callable/function!")
            raise TypeError
        try:
            test_vec = np.random.randn(size)
            log_density(test_vec)
        except:
            print(f"Failed to evaluate the log-density using a randomly generated vector")
            raise TypeError

        # Log-Density gradient
        log_density_grad = self._CONFIGURATIONS['log_density_grad']
        if log_density_grad is None:
            log_density_grad = self._create_func_grad(log_density, size=size)

        elif not callable(log_density_grad):
            print(f"The 'log_density_grad' found in the configurations is not a valid callable/function!")
            raise TypeError

        try:
            test_vec = np.random.randn(size)
            grad = log_density_grad(test_vec)
            assert grad.size == test_vec.size, ""
        except:
            print(f"Failed to evaluate the log-density using a randomly generated vector")
            raise TypeError

        self._LOG_DENSITY = log_density
        self._LOG_DENSITY_GRAD = log_density_grad

    def potential_energy(self, state, verbose=False):
        """
        Evaluate the value of the potential energy at the given `state`
            The potential energy is the negative value of the logarithm of
            the unscaled posterior density function
        """
        if np.any(np.isnan(state)):
            if verbose:
                print("NaN values in the passed state")
                print(f"Received State:\n {repr(state)}")
            # raise ValueError
            return np.nan
        return -self.log_density(state)

    def potential_energy_grad(self, state, verbose=False):
        """
        Evaluate the gradient of the potential energy at the given `state`
            The potential energy is the negative value of the logarithm of
            the unscaled posterior density function
        """
        if np.any(np.isnan(state)):
            if verbose:
                print("NaN values in the passed state")
                print(f"Received State:\n {repr(state)}")
            # raise ValueError
            return np.nan
        return -self.log_density_grad(state)

    def kinetic_energy(self, momentum):
        """
        Evaluate the Kinetic energy of the posterior; this is independent from the state
            and is evaluated as the weighted l2 norm of the momentum
            (scaled by the inverse of hte mass matrix);
            This is half of the squared Mahalanobis distance of the Gaussian momentum

        :raises:
            - :py:class:`TypeError` is raised if the passed momentum has invalid shape/type/size
        """
        momentum = np.asarray(momentum).flatten()
        if momentum.size != self._CONFIGURATIONS['size']:
            print(f"The passed momentum has invalid size;"
                  f"received {momentum}, expected {self._CONFIGURATIONS['size']}")
            raise TypeError

        return 0.5 * np.dot(momentum, self.mass_matrix_inv_matvec(momentum))

    def total_Hamiltonian(self, momentum, state):
        """
        Evaluate the value of the total energy function:
            Hamiltonian = kinetic energy + potential energy
        """
        return self.kinetic_energy(momentum) + self.potential_energy(state)

    def build_Hamiltonian_trajectory(self, momentum, state,
                                     step_size=_DEF_CONFIGURATIONS['symplectic_integrator_stepsize'],
                                     num_steps=_DEF_CONFIGURATIONS['symplectic_integrator_num_steps'],
                                     randomize_step_size=False):
        """
        Given the current momentum and state pair of the Hamiltonian system, generate a trajectory
            of (momentum, state).
        :param momentum:
        :param state:
        :param bool randomize_step_size: if `True` a tiny random number is added to the passed step
            size to help improve space exploration
        """
        # local copies
        momentum = np.asarray(momentum).flatten()
        state    = np.asarray(state).flatten()

        trajectory = [(momentum, state)]
        # Loop over number of steps, for each step update current momentum and state then append to trajectory
        for _ in range(num_steps):
            trajectory.append(
                apply_symplectic_integration(
                momentum=trajectory[-1][0],
                state=trajectory[-1][1],
                )
            )
        return trajectory

    def apply_symplectic_integration(self,
                                     momentum,
                                     state,
                                     step_size=_DEF_CONFIGURATIONS['symplectic_integrator_stepsize'],
                                     num_steps=_DEF_CONFIGURATIONS['symplectic_integrator_num_steps'],
                                     randomize_step_size=False,
                                     symplectic_integrator='3-stage',
                                     ):
        """
        Apply one full step of size `step_size` of the symplectic integrator to the Hamiltonian system

        :parm momentum:
        :param state:
        :param int num_steps:
        :param float step_size:
        :param str symplectic_integrator: name of the symplectic integrator to use;
            acceptable are: 'verlet', 'leapfrog', '2-stage', '3-stage',
                where both 'leapfrog' and 'verlet' are equivalent
        :param bool randomize_step_size: if `True` a tiny random number is added to the passed step
            size to help improve space exploration

        :returns: a tuple (p, s) where p and s are the integrated (forward in time) momentum and state respectively
        """
        # local copies
        if np.any(np.isnan(momentum)):
            print("Cannot apply symplectic integorator; NaN values found in the passed momentum")
            raise ValueError
        if np.any(np.isnan(state)):
            print("Cannot apply symplectic integorator; NaN values found in the passed state")
            raise ValueError
        current_momentum = np.asarray(momentum).flatten()
        current_state    = np.asarray(state).flatten()

        state_space_dimension = self._CONFIGURATIONS['size']
        if not (current_momentum.size == current_state.size == state_space_dimension):
            print(f"The momentum and state must be of the same size as the underlying space dimnsion; "
                  f"State size: {current_state.size}, "
                  f"Momentum size: {current_momentum.size}, "
                  f"Underlying space dimension: {state_space_dimension}")
            raise TypeError

        # validate step size (and randomize if asked)
        if step_size <= 0:
            print(f"Step size of the symplectic integrator must be positive!!")
            raise ValueError

        if randomize_step_size:
            # random step size perturbation (update random state)
            np_state = np.random.get_state()
            np.random.set_state(self.random_state)
            u = (np.random.rand() - 0.5) * 0.4  # perturb step-size:
            self.random_state = np.random.get_state()
            np.random.set_state(np_state)

            h = (1 + u) * step_size
        else:
            h = step_size

        for _ in range(num_steps):
            #
            if re.match(r"\A(verlet|leapfrog)\Z", symplectic_integrator, re.IGNORECASE):

                # Update state
                proposed_state = current_state + (0.5*h) * self.mass_matrix_inv_matvec(current_momentum)
                # print("1: proposed state", proposed_state)

                # Update momentum
                grad = self.potential_energy_grad(proposed_state)
                proposed_momentum = current_momentum - h * grad
                # print("<: proposed momentum", proposed_momentum)

                # Update state again
                proposed_state += (0.5*h) * self.mass_matrix_inv_matvec(proposed_momentum)

            elif re.match(r"\A2(-|_| )*stage(s)*\Z", symplectic_integrator, re.IGNORECASE):
                a1 = 0.21132
                a2 = 1.0 - 2.0 * a1
                b1 = 0.5

                proposed_state = current_state + (a1*h) * self.mass_matrix_inv_matvec(current_momentum)

                grad = self.potential_energy_grad(proposed_state)
                proposed_momentum = current_momentum - (b1*h) * grad

                proposed_state = proposed_state + (a2*h) * self.mass_matrix_inv_matvec(proposed_momentum)

                grad = self.potential_energy_grad(proposed_state)
                proposed_momentum = proposed_momentum - (b1*h) * grad

                proposed_state += (a1*h) * self.mass_matrix_inv_matvec(proposed_momentum)

            elif re.match(r"\A3(-|_| )*stage(s)*\Z", symplectic_integrator, re.IGNORECASE):
                a1 = 0.11888010966548
                a2 = 0.5 - a1
                b1 = 0.29619504261126
                b2 = 1.0 - 2.0 * b1

                proposed_state = current_state + (a1*h) * self.mass_matrix_inv_matvec(current_momentum)

                grad = self.potential_energy_grad(proposed_state)
                proposed_momentum = current_momentum - (b1*h) * grad

                proposed_state = proposed_state + (a2*h) * self.mass_matrix_inv_matvec(proposed_momentum)

                grad = self.potential_energy_grad(proposed_state)
                proposed_momentum = proposed_momentum - (b2*h) * grad

                proposed_state = proposed_state + (a2*h) * self.mass_matrix_inv_matvec(proposed_momentum)

                grad = self.potential_energy_grad(proposed_state)
                proposed_momentum = proposed_momentum - (b1*h) * grad

                proposed_state += (a1*h) * self.mass_matrix_inv_matvec(proposed_momentum)

            else:
                raise ValueError("Unsupported symplectic integrator %s" % symplectic_integrator)

            # Update current state and momentum
            current_momentum = proposed_momentum
            current_state    = proposed_state

        return (proposed_momentum, proposed_state)

    def start_MCMC_sampling(self, sample_size, initial_state=None, randomize_step_size=False,
                           full_diagnostics=False, verbose=False, ):
        """
        Start the HMC sampling procedure with initial state as passed.
        Use the underlying configurations for configuring the Hamiltonian trajectory, burn-in and mixin settings.

        :param int sample_size: number of smaple points to generate/collect from the predefined target distribution
        :param initial_state: initial point of the chain (any point that falls in the target distribution or near by it
            will result in faster convergence). You can try prior mean if this is used in a Bayesian approach
        :param bool randomize_step_size: if `True` a tiny random number is added to the passed step
            size to help improve space exploration
        :param bool full_diagnostics: if `True` all generated states will be tracked and kept for full disgnostics, otherwise,
            only collected samples are kept in memory
        :param bool verbose: screen verbosity
        """

        # Extract configurations from the configurations dictionary
        state_space_dimension = self._CONFIGURATIONS['size']
        burn_in_steps         = self._CONFIGURATIONS['burn_in']
        mixing_steps          = self._CONFIGURATIONS['mix_in']
        symplectic_integrator = self._CONFIGURATIONS['symplectic_integrator']
        hamiltonian_step_size = self._CONFIGURATIONS['symplectic_integrator_stepsize']
        hamiltonian_num_steps = self._CONFIGURATIONS['symplectic_integrator_num_steps']
        constraint_test       = self._CONFIGURATIONS['constraint_test']

        liner, sliner = '=' * 53, '-' * 40
        if verbose:
            print("\n%s\nStarted Sampling\n%s\n" % (liner, liner))

        # Chain initial state
        if initial_state is None:
            initial_state = self.generate_white_noise(state_space_dimension)
        else:
            initial_state = np.array(initial_state).flatten()
            if initial_state.size != state_space_dimension:
                print(f"Passed initial stae has invalid shape/size"
                      f"Passed initial state has size {initial_state.size}"
                      f"Expected size: {state_space_dimension}")
                raise TypeError

        # Setup and construct the chain using HMC proposal:
        chain_length = burn_in_steps + sample_size * mixing_steps

        # Initialize the chain
        current_state = initial_state.copy()  # initial state = ensemble mean

        # All generated sample points will be kept for testing and efficiency analysis
        chain_state_repository    = [initial_state]
        proposals_repository      = []
        acceptance_flags          = []
        acceptance_probabilities  = []
        uniform_random_numbers    = []
        collected_ensemble        = []
        map_estimate             = None
        map_estimate_log_density = -np.infty

        # Build the Markov chain
        start_time = time.time()  # start timing
        for chain_ind in range(chain_length):

            ## Proposal step :propose (momentum, state) pair
            # Generate a momentum proposal
            current_momentum = self.generate_white_noise(size=state_space_dimension)
            current_momentum = self.mass_matrix_sqrt_matvec(current_momentum)

            # Advance the current state and momentum to propose a new pair:
            proposed_momentum, proposed_state = self.apply_symplectic_integration(
                momentum=current_momentum,
                state=current_state,
                num_steps=hamiltonian_num_steps,
                step_size=hamiltonian_step_size,
                randomize_step_size=randomize_step_size,
                symplectic_integrator=symplectic_integrator,
            )

            # print("proposed_momentum, proposed_state", proposed_momentum, proposed_state)

            ## MH step (Accept/Reject) proposed (momentum, state)
            # Calculate acceptance proabability
            # Total energy (Hamiltonian) of the extended pair (proposed_momentum,
            # Here, we evaluate the kernel of the posterior at both the current and the proposed state proposed_state)
            current_energy  = self.total_Hamiltonian(momentum=current_momentum, state=current_state)
            constraint_violated = False
            if constraint_test is not None:
                if not constraint_test(proposed_state): constraint_violated = True

            if constraint_violated:
                acceptance_probability = 0

            else:
                proposal_kinetic_energy   = self.kinetic_energy(proposed_momentum)
                proposal_potential_energy = self.potential_energy(proposed_state)
                proposal_energy           = proposal_kinetic_energy + proposal_potential_energy

                energy_loss = proposal_energy - current_energy
                _loss_thresh = 1000
                if abs(energy_loss) >= _loss_thresh:  # this should avoid overflow errors
                    if energy_loss < 0:
                        sign = -1
                    else:
                        sign = 1
                    energy_loss = sign * _loss_thresh
                acceptance_probability = np.exp(-energy_loss)
                acceptance_probability = min(acceptance_probability, 1.0)

                # Update Mode (Map Point Estimate)
                if - proposal_potential_energy > map_estimate_log_density:
                    map_estimate             = proposed_state.copy()
                    map_estimate_log_density = - proposal_potential_energy

            # a uniform random number between 0 and 1
            np_state = np.random.get_state()
            np.random.set_state(self.random_state)
            uniform_probability = np.random.rand()
            self.random_state = np.random.get_state()
            np.random.set_state(np_state)

            # MH-rule
            if acceptance_probability > uniform_probability:
                current_state   = proposed_state
                accept_proposal = True
            else:
                accept_proposal = False

            if verbose:
                print(f"\rHMC Iteration [{chain_ind+1:4d}/{chain_length:4d}]; Accept Prob: {acceptance_probability:3.2f}; --> Accepted? {accept_proposal}", end="  ")

            #
            if chain_ind >= burn_in_steps and chain_ind % mixing_steps==0:
                collected_ensemble.append(current_state.copy())

            # Update Results Repositories:
            if full_diagnostics:
                proposals_repository.append(proposed_state)
                acceptance_probabilities.append(acceptance_probability)
                uniform_random_numbers.append(uniform_probability)
                #
                if accept_proposal:
                    acceptance_flags.append(1)
                else:
                    acceptance_flags.append(0)
                chain_state_repository.append(np.squeeze(current_state))

        # Stop timing
        chain_time = time.time() - start_time

        # ------------------------------------------------------------------------------------------------

        # Now output diagnostics and show some plots :)
        if full_diagnostics:
            chain_diagnostics = self.mcmc_chain_diagnostic_statistics(
                proposals_repository=proposals_repository,
                chain_state_repository=chain_state_repository,
                collected_ensemble=collected_ensemble,
                acceptance_probabilities=acceptance_probabilities,
                uniform_probabilities=uniform_random_numbers,
                acceptance_flags=acceptance_flags,
                map_estimate=map_estimate,
            )

        #
        # ======================================================================================================== #
        #                Output sampling diagnostics and plot the results for 1 and 2 dimensions                   #
        # ======================================================================================================== #
        #
        if verbose:
            print("MCMC sampler:")
            print(f"Time Elapsed for MCMC sampling: {chain_time} seconds")
            print(f"Acceptance Rate: {chain_diagnostics['acceptance_rate']:.2f}")

        sampling_results = dict(
            chain_state_repository=chain_state_repository,
            collected_ensemble=collected_ensemble,
            proposals_repository=proposals_repository,
            acceptance_flags=acceptance_flags,
            acceptance_probabilities=acceptance_probabilities,
            uniform_random_numbers=uniform_random_numbers,
            chain_diagnostics=chain_diagnostics,
            map_estimate=map_estimate,
            map_estimate_log_density=map_estimate_log_density,
            chain_time=chain_time,
        )
        return sampling_results

    def mcmc_chain_diagnostic_statistics(self,
                                         proposals_repository,
                                         chain_state_repository,
                                         uniform_probabilities,
                                         acceptance_probabilities,
                                         collected_ensemble,
                                         map_estimate=None,
                                         acceptance_flags=None,
                                         ):
        """
        Return diagnostic statistics of the chain such as the rejection rate, acceptance ratio, etc.
        """
        if acceptance_flags is None:
            acceptance_flags = np.asarray(acceptance_probabilities >= uniform_probabilities, dtype=np.int)
        else:
            acceptance_flags = np.asarray(acceptance_flags)
        acceptance_rate = float(acceptance_flags.sum()) / np.size(acceptance_flags) * 100.0
        rejection_rate = (100.0 - acceptance_rate)

        # Plots & autocorrelation, etc.
        plot_mcmc_results_nd(
            collected_ensemble,
            log_density=self.log_density,
            map_estimate=map_estimate,
            title="HMC",
            filename_prefix="HMC_Sampling",
        )
        # TODO: Add More; e.g., effective sample size, etc.

        # Return all diagonistics in a dictionary
        chain_diagnositics = dict(
            acceptance_rate=acceptance_rate,
            rejection_rate=rejection_rate,
        )

        return chain_diagnositics

    @property
    def random_state(self):
        """Get a handle of the current internal random state"""
        return self._RANDOM_STATE
    @random_state.setter
    def random_state(self, value):
        """Update the internal random state"""
        try:
            np_state = np.random.get_state()
            np.random.set_state(value)
            self._RANDOM_STATE = value
            np.random.set_state(np_state)
        except:
            print("Invalid random state passed of type '{0}'".format(type(value)))
            raise TypeError


# General functions supporting Random sampling; e.g., plotting, etc.
def plot_mcmc_results_2d(collected_ensemble,
                         log_density=None,
                         map_estimate=None,
                         xlim=None,  # (-6, 6),
                         ylim=None,  # (-2, 11),
                         title=None,
                         linewidth=1.0,
                         markersize=3,
                         fontsize=18,
                         fontweight='bold',
                         keep_plots=False,
                         filename_prefix="MCMC_Sampling",
                         verbose=False,
                         ):
    """
    """
    utility.plots_enhancer(fontsize=fontsize,
                           fontweight=fontweight,
                           usetex=True,
                           )
    if verbose:
        print("*** Creating 2D plots ***")

    if log_density is None:
        evaluate_pdf = None
    else:
        evaluate_pdf = lambda x: np.exp(log_density(x))

    # Plot resutls:
    actual_ens = np.asarray(collected_ensemble)
    sample_size = np.size(actual_ens, 0)

    chain_state_repository = actual_ens.copy()

    if verbose:
        print("\n Constructing plot information... ")

    # plot 1: data + best-fit mixture
    if xlim is not None:
        x_min, x_max = xlim
    else:
        x_min = np.min(actual_ens, axis=0)[0]
        x_max = np.max(actual_ens, axis=0)[0]
        offset = abs(x_max-x_min) * 0.1
        x_min -= offset
        x_max += offset

    if ylim is not None:
        y_min, y_max = ylim
    else:
        y_min = np.min(actual_ens, axis=0)[1]
        y_max = np.max(actual_ens, axis=0)[1]
        offset = abs(y_max-y_min) * 0.1
        y_min -= offset
        y_max += offset

    x_size = y_size = 200
    x = np.linspace(x_min, x_max, x_size)
    y = np.linspace(y_min, y_max, y_size)
    x, y = np.meshgrid(x, y)


    posterior_pdf_vals = np.empty((x_size, y_size))
    posterior_pdf_vals[...] = np.nan
    if evaluate_pdf is not None:
        for i in range(x_size):
            for j in range(y_size):
                state_tmp = np.array([x[i][j], y[i][j]])
                posterior_pdf_vals[i,j] = evaluate_pdf(state_tmp)

    z_min = np.max(posterior_pdf_vals) - 0.01
    z_max = np.max(posterior_pdf_vals) + 0.03

    # mask:
    posterior_pdf_vals_cp = posterior_pdf_vals.copy()

    # Animate the sampler steps:
    fig1, ax = plt.subplots(facecolor='white')
    ax.contour(x, y, posterior_pdf_vals_cp, colors='k')
    CS = ax.contourf(x, y, posterior_pdf_vals_cp, 14, cmap="RdBu_r")
    line, = ax.plot(chain_state_repository[0, 0], chain_state_repository[0, 1], '-r', linewidth=linewidth, markersize=markersize, alpha=0.75)
    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(x))
        return line,
    def animate(frame_no):
        data = chain_state_repository[: frame_no+1, :]
        # ax.clear()
        line.set_xdata(data[:, 0])
        line.set_ydata(data[:, 1])
        line.set_linewidth(linewidth)
        line.set_color('red')
        ax.scatter(data[-1, 0], data[-1, 1], alpha=0.75, s=15)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.set_title("$Iteration:%04d$" % (frame_no+1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    save_count=np.size(chain_state_repository, 0)
    # save_count = 250
    ani = animation.FuncAnimation(fig1, animate, interval=100, save_count=save_count)
    if title is not None: fig1.suptitle(title + " Sampling Animation")
    filename = f"{filename_prefix}_diagnostics.mp4"
    ani.save(filename, dpi=900)
    print(f"Saved plot to {filename}")
    if not keep_plots: plt.close(fig1)

    #
    # Plot prior, likelihood, posterior, and histogram
    fig2 = plt.figure(figsize=(16, 5), facecolor='white')
    # plot contuour of the posterior and scatter of the ensemble
    ax1 = fig2.add_subplot(1, 3, 1)
    # ax1.set_xticks(np.arange(x_min, x_max, 2))
    # ax1.set_xticklabels(np.arange(x_min, x_max, 2))
    ax1.set_xlabel("$x$", fontsize=fontsize)
    ax1.set_ylabel("$y$", fontsize=fontsize)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    CS1 = ax1.contour(x, y, posterior_pdf_vals)
    ax1.scatter(actual_ens[:, 0], actual_ens[:, 1], alpha=0.75, s=15)

    # Add map estimate if passed:
    print("MAP Estimate...", map_estimate)
    if map_estimate is not None:
        ax1.scatter(map_estimate[0], map_estimate[1], marker='^', alpha=0.65, s=45, label='MAP')
    ax1.set_aspect('auto')

    #
    ax2 = fig2.add_subplot(1, 3, 2)
    # ax2.set_xticks(np.arange(x_min, x_max, 2))
    # ax2.set_xticklabels(np.arange(x_min, x_max, 2))
    ax2.set_xlabel("$x$", fontsize=fontsize)
    ax2.set_ylabel("$y$", fontsize=fontsize)
    # ax2.set_xlim(x_min, x_max)
    # ax2.set_ylim(y_min, y_max)
    ax2.contour(x, y, posterior_pdf_vals_cp, linewidth=0.5, colors='k')
    CS2 = ax2.contourf(x, y, posterior_pdf_vals_cp, 14, cmap="RdBu_r", alpha=0.85)
    ax2.plot(chain_state_repository[:, 0], chain_state_repository[:, 1], '-ro', linewidth=linewidth, markersize=markersize, alpha=0.15)
    if map_estimate is not None:
        ax2.scatter(map_estimate[0], map_estimate[1], marker='^', alpha=0.65, s=45, label='MAP')
    ax2.set_aspect('auto')

    #
    # Plot autocorrelation:
    ax3 = fig2.add_subplot(1, 3, 3)
    if autocorrelation_plot is None:
        print("pandas is not installed; autocorrelation plot is not generated")
    else:
        autocorrelation_plot(posterior_pdf_vals, ax=ax3)
        ax3.set_aspect('auto')

    if title is not None: fig2.suptitle(title + " Diagnostics")
    #
    filename = f"{filename_prefix}_diagnostics.pdf"
    fig2.savefig(filename, dpi=900, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"Saved plot to {filename}")
    if not keep_plots: plt.close(fig2)

def plot_mcmc_results_nd(collected_ensemble,
                         log_density=None,
                         map_estimate=None,
                         title=None,
                         labels=None,
                         fontsize=18,
                         fontweight='bold',
                         keep_plots=False,
                         filename_prefix="MCMC_Sampling",
                         verbose=False,
                         ):
    """
    Create plots for MCMC results
    """
    utility.plots_enhancer(fontsize=fontsize,
                           fontweight=fontweight,
                           usetex=True,
                           )
    if verbose:
        print("*** Creating bivariate plots array ***")

    # Plot resutls:
    actual_ens = np.asarray(collected_ensemble)
    sample_size, nvars = actual_ens.shape

    if labels is None:
        columns = [f'X{i+1}' for i in range(nvars)]
    else:
        columns = [l for l in labels]

    # Create dataframe object and plot
    if sns is None:
        print("seaborn is not installed; pair plots are not generated")
    elif pd is None:
        print("pandas is not installed; pair plots are not generated")
    else:
        df = pd.DataFrame(actual_ens, columns=columns, )
        g = sns.pairplot(df, diag_kind="kde")
        g.map_lower(sns.kdeplot, levels=5, color=".4")
        fig = g.figure
        if title is not None: fig.suptitle(title + " PairPlot")

        # save to file
        filename = f"{filename_prefix}_PairPlot.pdf"
        g.savefig(filename, dpi=900, bbox_inches='tight', facecolor='white', format='pdf')
        print(f"Saved plot to {filename}")
        if not keep_plots:
            plt.close(fig)

    # plot_mcmc_results_2d(sample)
    if nvars == 2:
        plot_mcmc_results_2d(actual_ens,
                             log_density=log_density,
                             map_estimate=map_estimate,
                             title=title,
                             fontsize=fontsize,
                             fontweight=fontweight,
                             keep_plots=keep_plots,
                             filename_prefix=filename_prefix,
                             verbose=verbose,
                             )


## Simple interfaces (to generate instances from classes developed here).
def create_Gaussian_proposal(size,
                             mean=None,
                             variance=1,
                             random_seed=None,
                       ):
    """
    Given the size of the target space, create and return an :py:class:`GaussianProposal` instance/object
        to generate samples using Gaussian proposal centered around current state (or predefined mean)
        Configurations/settings can be updated after inistantiation

    This function shows how to create :py:class:`GaussianProposal` instances
        (with some or all configurations passed)
    """
    configs = dict(
        size=size,
        mean=mean,
        variance=variance,
        random_seed=random_seed,
    )
    return GaussianProposal(configs)

def create_mcmc_sampler(size,
                       log_density,
                       burn_in=100,
                       mix_in=10,
                       constraint_test=None,
                       random_seed=None,
                       ):
    """
    Given the size of the target space, and a function to evalute log density,
        create and return an :py:class:`MCMCSampler` instance/object to generate samples
        using standard MCMC sampling approach.
        Configurations/settings can be updated after inistantiation

    This function shows how to create :py:class:`MCMCSampler` instances (with some or all configurations passed)
    """
    configs = dict(
        size=size,
        log_density=log_density,
        burn_in=burn_in,
        mix_in=mix_in,
        constraint_test=constraint_test,
        random_seed=random_seed,
    )
    return MCMCSampler(configs)


def create_hmc_sampler(size,
                       log_density,
                       log_density_grad=None,
                       burn_in=100,
                       mix_in=5,
                       symplectic_integrator='verlet',
                       symplectic_integrator_stepsize=1e-2,
                       symplectic_integrator_num_steps=10,
                       mass_matrix=1,
                       constraint_test=None,
                       random_seed=None,
                       ):
    """
    Given the size of the target space, and a function to evalute log density,
        create and return an :py:class:`HMCSampler` instance/object to generate samples using HMC sampling approach.
        Configurations/settings can be updated after inistantiation

    This function shows how to create :py:class:`HMCSampler` instances (with some or all configurations passed)
    """
    configs = dict(
        size=size,
        log_density=log_density,
        log_density_grad=log_density_grad,
        burn_in=burn_in,
        mix_in=mix_in,
        symplectic_integrator=symplectic_integrator,
        symplectic_integrator_stepsize=symplectic_integrator_stepsize,
        symplectic_integrator_num_steps=symplectic_integrator_num_steps,
        mass_matrix=mass_matrix,
        constraint_test=constraint_test,
        random_seed=random_seed,
    )
    return HMCSampler(configs)


def banana_potential_energy_value(state, a=2.15, b=0.75, rho=0.9, ):
    """
    Potential energy of the posterir. This is dependent on the target state, not the momentum.
    It is the negative the posterior-log, and MUST be implemented for each distribution
    """
    x, y = state[:]
    #
    pdf_val = 1 / (2 * (1 - rho**2))
    t1 = x**2 / a**2 + a**2 * (y - b * x**2 / a**2 - b * a**2)**2
    t2 = - 2 * rho * x * (y - b * x**2 / a**2 - b * a**2)
    pdf_val = (t1 + t2) / (2 * (1 - rho**2))
    #
    return pdf_val

def banana_potential_energy_gradient(state, a=2.15, b=0.75, rho=0.9, ):
    """
    Gradient of the Potential energy of the posterir.
    """
    x, y = state.flatten()
    #
    pdf_grad = np.empty(2)
    pdf_grad[:] = 1 / (2 * (1-rho**2))
    #
    t1_x = 2 * x / a**2 + 2 * a**2 * (y - b * x**2 / a**2 - b * a**2) * (-2 * b * x / a**2)
    t1_y = 2 * a**2 * (y - b * x**2 / a**2 - b * a**2)
    t1 = np.array([t1_x, t1_y])
    #
    t2_x = - 2 * rho * y + 6 * b * rho * x**2 / a**2 + 2 * rho * b * a**2

    t2_y = - 2 * rho * x
    t2 = np.array([t2_x, t2_y])
    #
    pdf_grad *= (t1 + t2)
    #
    return pdf_grad

def sample_banana_distribution(sample_size, verbose=False, ):
    """  """
    initial_state    = [0, 0]
    log_density      = lambda x: -banana_potential_energy_value(x)
    log_density_grad = lambda x: -banana_potential_energy_gradient(x)

    sampler = create_hmc_sampler(
        size=2,
        log_density=log_density,
        log_density_grad=log_density_grad,
    )

    # Sample
    hmc_results = sampler.start_MCMC_sampling(sample_size=sample_size,
                                              initial_state=initial_state,
                                              full_diagnostics=True,
                                              )
    sample       = hmc_results['collected_ensemble']
    map_estimate = hmc_results['map_estimate']

    return sample, map_estimate


