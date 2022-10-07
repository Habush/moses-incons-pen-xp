from notebooks.variable_selection.MosesEstimator import *
from notebooks.variable_selection.mixed_hmc_jax import *

import numba
from sklearn.metrics import mutual_info_score
import numpyro as npyro
from numpyro.infer import MCMC, MixedHMC, HMC
import numpyro.distributions as dist
import numpy as np
import jax
import jax.numpy as jnp
import distrax

def samples_to_feats(samples):
    feats = []
    for i in range(samples.shape[0]):
        feat_idx = list(np.nonzero(samples[i])[0])
        if len(feat_idx) == 0 or feat_idx in feats: continue
        else: feats.append(feat_idx)
    return feats

def rank_by_mi(samples, J, X, Y, eta=1.0, mu=1.0, B=1.0):
    samples_c = samples.reshape((samples.shape[0]*samples.shape[1], samples.shape[2]))
    samples_unq = np.unique(samples_c, axis=0)
    energy_vals = [bmm_energy(samples_unq[i], J, eta=eta, mu=1.0) for i in range(samples_unq.shape[0])]
    feats = samples_to_feats(samples_unq)
    mi_scores = []
    for eng, feat in zip(energy_vals, feats):
        s_score = 0.0
        for f in feat:
            mi = mutual_info_score(Y, X[:,f])
            s_score += mi

        mi_scores.append(s_score - B*eng)

    idxs = np.argsort(np.array(mi_scores))[::-1]
    feats = list(np.array(feats, dtype=object)[idxs])

    return feats, np.sort(np.array(mi_scores))[::-1]

def rank_by_cond_prob(samples, J, X_train, y_train, energy_fn, eta=1.0):
    samples_c = samples.reshape((samples.shape[0]*samples.shape[1], samples.shape[2]))
    samples_unq = np.unique(samples_c, axis=0)
    prob_vals = [np.exp(-energy_fn(samples_unq[i], J, X=X_train, Y=y_train, eta=eta)) for i in range(samples_unq.shape[0])]
    idx = np.argsort(np.array(prob_vals))[::-1]
    samples_unq = samples_unq[idx]
    feats = samples_to_feats(samples_unq)
    return feats

def rank_hmc_feats_rand(samples, n=100):
    samples_unq = np.unique(samples, axis=0)
    choice_idx = np.random.choice(samples_unq.shape[0], size=n)
    gamma_choice = samples_unq[choice_idx]
    feats_choice = samples_to_feats(gamma_choice)

    return feats_choice

def rank_hmc_feats_mi(gammas, betas, X, y, J, sigma, eta, mu, B, loc=None):
    potential_fn = generate_potential_energy_fn(X, y, J, sigma, eta, mu, loc)

    gammas_unq, unq_idx = np.unique(gammas, axis=0, return_index=True)
    betas_unq = betas[unq_idx]

    gamma_eng = [potential_fn(gammas_unq[i], betas_unq[i]) for i in range(gammas_unq.shape[0])]
    feats = samples_to_feats(gammas_unq)

    mi_scores = []
    for eng, feat in zip(gamma_eng, feats):
        score = 0.0
        for f in feat:
            mi = mutual_info_score(y, X[:,f])
            score += mi

        mi_scores.append(score - B*eng)

    idxs = np.argsort(np.array(mi_scores))[::-1]

    feats = samples_to_feats(gammas_unq)

    feats = list(np.array(feats)[idxs])

    return feats, np.sort(np.array(mi_scores))[::-1]

def rank_hmc_feats_eng(gammas, betas, X, y, J, eta, mu):
    sigma = np.cov(betas, rowvar=False)
    loc = np.mean(betas, axis=0)
    k = gammas.shape[0]
    potential_fn = generate_potential_energy_fn(X, y, J, sigma, eta, mu, loc)
    sample_engs = np.zeros(k)
    for i in range(k):
        sample_engs[i] = potential_fn(gammas[i], betas[i])


    sort_idx = np.argsort(sample_engs)
    sort_eng = sample_engs[sort_idx]

    return sort_idx, sort_eng

def get_rand_feats(p, n=100):
    feats = []
    for i in range(n):
        idx = np.random.randint(0, 2, size=p)
        feats.append(list(np.nonzero(idx)[0]))

    return feats

def logistic(x):
    return 1/(1 + jnp.exp(-x))

def jax_prng_key():
    return jax.random.PRNGKey(np.random.randint(int(1e5)))

def hamm(a,b):
    return len(np.nonzero(a != b)[0])

def average_length(a):
    s = []
    for i in a:
        s.append(len(i))

    return np.mean(np.array(s))
def average_length_np(a):
    k = a.shape[0]
    s = np.zeros(k)
    for i in range(k):
        s[i] = np.count_nonzero(a[i])

    return np.mean(s)

def average_hamm_dist(feats, seed_idx, feats_sample, p):
    feats_true_idx = np.array(feats[seed_idx]) - 1
    feats_true = np.zeros(p)
    feats_true[feats_true_idx] = 1

    hamm_dist = []

    for i in range(len(feats_sample)):
        feats_s = np.zeros(p)
        feats_s[np.array(feats_sample[i])] = 1
        hamm_dist.append(hamm(feats_true, feats_s))

    return np.mean(np.array(hamm_dist), axis=0)

@numba.njit(nogil=True)
def quad_prod(x, A):
    return ((x.T @ A) @ x)

@numba.njit(nogil=True)
def bmm_energy(gamma, G, X=None, Y=None, eta=1.0, mu=1.0):
    return eta*0.5*-quad_prod(gamma, G) + mu*np.sum(gamma)

@numba.njit(nogil=True)
def bmm_energy_cond(gamma, G, X=None, Y=None, eta=1.0, mu=1.0):

    return eta*0.5*-quad_prod(gamma, G) + np.sum(gamma) + mu*np.sum((1 - Y) * (X @ np.diag(gamma)))


@numba.njit(nogil=True)
def metropolis(J, times, energy_fn, X=None, Y=None, c=5, n=100, eta=1.0, mu=1.0):
    net_spins = np.zeros((c, times))
    net_energy = np.zeros((c, times))
    N = J.shape[0]
    samples = np.zeros((c, n, N))
    for j in range(c):
        init_state = np.random.randint(0, 2, size=N)
        init_state = init_state.astype(np.float_)
        energy = energy_fn(init_state, J, X, Y, eta=eta, mu=mu)
        curr_st = init_state.copy()
        for t in range(0,times):
            # 2. pick random point on array and flip spin
            i = np.random.randint(0, N)
            prev_st = curr_st.copy()
            E_prev = energy_fn(prev_st, J, X, Y, eta=eta, mu=mu)
            val = np.abs(curr_st[i] - 1) # flip the current position
            # val = prev_st[i]*-1
            prev_st[i] = val
            # compute change in energy
            E_curr = energy_fn(prev_st, J, X, Y, eta=eta, mu=mu)
            # 3 / 4. change state with designated probabilities
            dE = E_curr-E_prev
            prob = np.exp(-dE)

            if (dE>0)*(np.random.random() < prob): ## accept the move lower probability state with probability p
                curr_st[i] = val
                energy += dE
            elif dE<=0: ##accept the move to higher probability state
                curr_st[i] = val
                energy += dE

            net_spins[j, t] = curr_st.sum()
            net_energy[j, t] = energy
            if times - t <= n:
                samples[j, n - (times - t)] = curr_st

    # samples = samples.reshape(c*n, N)

    return net_spins, net_energy, samples


def gamma_energy(gamma, J, eta, mu):
    return 0.5*eta*jnp.dot(jnp.dot(gamma.T, J), gamma) - mu*jnp.sum(gamma)

def model(X, y, sigma, J, eta=1.0, mu=1.0):
    beta = npyro.sample('beta', dist.MultivariateNormal(0, sigma))
    gamma = npyro.sample('gamma', dist.Bernoulli(np.full(X.shape[1], 0.5)))
    npyro.factor('gamma_lgp', gamma_energy(gamma, J, eta, mu))
    prob = npyro.deterministic("prob", logistic(jnp.dot(X, (beta * gamma))))
    # print(f"Probs: {prob.shape}")
    likelihood = npyro.sample("y", dist.Bernoulli(probs=prob),
                              obs=y)

def generate_potential_energy_fn(X, y, J, sigma, eta, mu, loc=None):
    X = jax.device_put(X)
    y = jax.device_put(y)
    J = jax.device_put(J)
    eta = jax.device_put(eta)
    mu = jax.device_put(mu)
    beta_dist = distrax.MultivariateNormalFullCovariance(loc, covariance_matrix=sigma)

    def potential_energy(gamma, beta):
        gamma_f = gamma.astype(jnp.float32)
        # beta_prior_potential = jnp.sum(
        #     0.5 * jnp.log(2 * jnp.pi * sigma ** 2) + 0.5 * beta ** 2 / sigma ** 2)
        beta_prior_potential = beta_dist.log_prob(beta)
        probs = 1 / (
                1 + jnp.exp(-jnp.dot(jnp.dot(X, jnp.diag(gamma_f)), beta))
        )
        likelihood_potential = -jnp.sum(
            y * jnp.log(probs + 1e-12) + (1 - y) * jnp.log(1 - probs + 1e-12)
        )

        gamma_potential = -0.5*eta*jnp.dot(jnp.dot(gamma_f.T, J), gamma_f) + mu*jnp.sum(gamma_f)
        # print(f"gamma eng: {gamma_potential}, beta eng: {beta_prior_potential}, likelihood eng: {likelihood_potential}")
        return beta_prior_potential + likelihood_potential + gamma_potential

    return potential_energy
