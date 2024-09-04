# %%
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import numpy as np
import zeus
import OptimizationProblem
import networkx as nx


from GraphicalModel import CausalGraphicalModel
from CausalModel import StructuralCausalModel
from CausalModel import CausalAssignmentModel


TRUE_THETAS = [-0.99, 0.99, 0.99]
TRUE_THETA_MAX = 1
TRUE_THETA_MIN = -1
interventions_num = 20


# Set some seed for reproducibility
np.random.seed(2024)


# Basically, we first sample the variable we are going to intervene upon (e.g., x1 or x2)
# then, we sample the value we want to add to that variable. We sample the value between -1 and 1.
interventions = []
for i in range(interventions_num):
    current_comparison = []
    node = str(np.random.choice(["x1", "x2"], 1)[0])  # Convert to Python string
    value = np.random.rand()
    # current_comparison.append([node, value])
    interventions.append([node, value])


# Interventions are simply a collection of group of tuples:
# [(node, value)]
print("intervention")
print(interventions)

simulation_scm = OptimizationProblem.generate_scm(TRUE_THETAS)
values_real_scm = []
for intervention in interventions:
    means_ = []
    scm_do_real = OptimizationProblem.intervened_data(simulation_scm, intervention, "soft")
    node, value = intervention

    mean_2 = np.mean(scm_do_real["x3"].values)
    if node == "x1":
        mean_1 = np.mean(scm_do_real["x2"].values)

    means_.append(mean_1)
    means_.append(mean_2)
    mean_1 = "no_value"
    values_real_scm.append(means_)

print(values_real_scm)

# %%
ndim = 3  # Number of non-zero parameters of the SCM matrix. In our case, it is 3
nwalkers = 50  # Number of particles (they call them walkers)
nsteps = 100  # Number of steps/iterations.

# Generally, the larger nwalkers and nsteps are, the longer the procedure will take.

# Increasing also the number of pairwise comparison makes everything even slower.
# I guess there are ways to speed up the likelihood function, but let's stick with this one for now.

# %%
# We need to sample an initial particle population. In order to do so, we need to be
# sure the initial particles are **valid** (e.g., they have a finite logprobability)
# Thus, we sample random particles from a uniform distribution and we compute the log
# probability. We only keep such particles that have a finite logprob.
#
# NOTE: if we increase the number of interventions, it would take much more time
# to find suitable particles/walkers!!!
print("start....")
start = []
while (len(start) < nwalkers):
    tmp = np.random.uniform(TRUE_THETA_MIN, TRUE_THETA_MAX, ndim)
    if np.isfinite(OptimizationProblem.log_posterior(tmp, interventions, values_real_scm,TRUE_THETA_MIN, TRUE_THETA_MAX )):
        print(f"FOUND {len(start) + 1}/{nwalkers}")
        start.append(tmp)


# %%
print("sampler....")
sampler = zeus.EnsembleSampler(nwalkers, ndim, OptimizationProblem.log_posterior, args=[interventions, values_real_scm, TRUE_THETA_MIN, TRUE_THETA_MAX],
                               verbose=True,
                               light_mode=True)  # Initialise the sampler
sampler.run_mcmc(start, nsteps)  # Run sampling
sampler.summary  # Print summary diagnostics

# %%
# Get the samples
samples = sampler.get_chain()

# Plot the walker trajectories for the first parameter of the 10
fig, ax = plt.subplots(3, 1, figsize=(16, 12))

# For each node, plot the convergence of the particles
for id in range(3):
    ax[id].plot(samples[:, :, id], alpha=0.5)
    ax[id].set_title(f"Convergence for $θ_{id + 1}$")
    ax[id].set_ylim(TRUE_THETA_MIN, TRUE_THETA_MAX)
plt.show()

# %%
# Given the samplers we compute the estimated theta and we compare everything with the correct distribution.
# In theory, if the procedure converged correctly, the plots should be very similar.

# We discard the first 50% of the chain particles, since that phase it is usually called burn-in.
chain = sampler.get_chain(flat=True, discard=0.5)

# %%
# We compute the average to compute the thetas
estimated_theta = np.mean(chain, axis=0)

# %%
print("estimated_theta")
print(estimated_theta)

# %%
print("TRUE_THETAS")
print(TRUE_THETAS)

# %%
# Sample elements from the estimated SCM
scm_estimated = OptimizationProblem.generate_scm(estimated_theta)
estm_data = scm_estimated.sample(1000)

# Get the true data
true_data = OptimizationProblem.simulation_scm.sample(1000)

# %%
# Plot the various distributions.
# We can see how they are basically dissimilar. Moreover, there is an error propagation effect.
# Since the estimation of X1 is partially correct (e.g., theta = -2.4 instead of 4), then all
# the subsequent X2 and X3 are poorly estimated.
# Interestingly, the preference tells us a different story.

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for id in range(3):
    sns.kdeplot(estm_data[f"x{id + 1}"], ax=ax[id])
    sns.kdeplot(true_data[f"x{id + 1}"], ls='--', ax=ax[id])
    ax[id].legend([f'$X_{id + 1}$', 'Truth']);