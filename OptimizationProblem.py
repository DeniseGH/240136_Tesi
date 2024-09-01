import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx

from GraphicalModel import CausalGraphicalModel
from CausalModel import StructuralCausalModel
from CausalModel import CausalAssignmentModel



# Here I changed the theta definition to be a vector instead of a matrix. Basically, since most of the entries
# of the matrix are zero, we can just consider a "flatter" representation.
def generate_scm(thetas):
    generator = np.random.default_rng(2024)
    scm = StructuralCausalModel({
        "x1": lambda n_samples, thetas=thetas: generator.normal(loc=0, scale=1, size=n_samples),
        "x2": lambda x1, n_samples, thetas=thetas: thetas[0] * x1 + generator.normal(loc=0, scale=1, size=n_samples),
        "x3": lambda x2, x1, n_samples, thetas=thetas: thetas[1] * x1 + thetas[2] * x2 + generator.normal(loc=0,
                                                                                                          scale=1,
                                                                                                      size=n_samples),
    })
    return scm


# %%
# I've changed this function to perform a "soft intervention" instead.
# Basically, given the intervention value v, we just add it to the variable (e.g., X = X + v).
# Previously, we were using a "hard intervention", which basically overwrite the value of the variable (e.g, X = v)
# by removing the corresponding parents edges.
# I had to change the StructuralCausalModel class (the sample() function) to enable soft interventions.
def intervened_data(scm, intervention, intervention_type):
    node, value = intervention
    # scm_do = scm.do(node)
    # We perform here the sampling of 100 instances to compute the expected value of the intervention E[X | do(X=X+v]
    ds_do = scm.sample(n_samples=100, set_values={node: np.full(100, value)}, type_of_intervention=intervention_type)
    return ds_do



# %%
# The prior is taken from zeus example. Basically, a particle is valid if and only
# if it lies between -5 and 5. If a particle k lies there, then P(\theta=k) = 1.
# Thus, the logprobability is 0, since np.log(1) = 0. However, if the particle
# is not in the correct range, the logprob is -infinity, since np.log(0) is undefined.
def log_prior(theta, a, b):
    if np.all(theta > a) and np.all(theta < b):
        return 0.0
    else:
        return -np.inf


# We consider a "hard" version of the likelihood function. Basically, given the user preferences
# we assign a positive probability only to those particles which matches **all** the ground truth choices.
# Basically, in the code below, estimated_result and ground_truth_choice must match always.
def likelihood(interventions, values_real_scm, estimated_thetas, epsilon=1):
    counter = []
    for intervention, scm_do_real in zip(interventions, values_real_scm):

        # generate the scm with the estimated thetas
        scm_temp = generate_scm(estimated_thetas)
        scm_do = intervened_data(scm_temp, intervention, "soft")

        node, value = intervention

        delta_2 = np.abs(np.mean(scm_do["x3"].values) - scm_do_real[1])

        result = True
        # If the node is "x1", also calculate the difference for x2
        if node == "x1":
            delta_1 = np.abs(np.mean(scm_do["x2"].values) - scm_do_real[0])

            # print("np.mean(delta_1)", np.mean(delta_1))
            # Check if all values in both delta_1 and delta_2 are less than 0.1
            if not ((delta_1 <= epsilon) and (delta_2 <= epsilon)):  # eventualmente si può mettere AND
                result = False

        else:
            # Check if all values in delta_2 are less than 0.1
            if not (delta_2 <= epsilon):
                result = False

        counter.append(result)

    # Calculate the likelihood based on the consistency count
    likelihood_value = np.all(counter)
    # print("likelihood_value", likelihood_value)
    return likelihood_value


# We combine the likelihood and the prior in the logposterior.
def log_posterior(thetas, interventions, values_real_scm, TRUE_THETA_MIN, TRUE_THETA_MAX):
    log_lk = likelihood(interventions, values_real_scm, thetas)
    log_lk = np.log(log_lk) if log_lk > 0 else -np.inf
    return log_lk + log_prior(thetas, TRUE_THETA_MIN, TRUE_THETA_MAX)