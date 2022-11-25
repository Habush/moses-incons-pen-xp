# Author Abdulrahman S. Omar<hsamireh@gmail>

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
import jax.numpy as jnp
import argparse
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn_genetic import GASearchCV
import numpy as np
from nn_util import *
from nn_models import *
from gibbs_sampler import *
from sgmcmc import *


def parse_args():

    parser = argparse.ArgumentParser(description="Run Neural Net experiments on Cancer Data.")

    parser.add_argument("--data", type=str, default=None, required=True, help="path to the dataset")
    parser.add_argument("--dir", type=str, default=None, required=True, help="directory to save results")
    parser.add_argument("--seed", type=int, default=None, required=True, help="seed to use for controlled randomness")
    parser.add_argument("--nfeats", type=int, default=70, required=True, help="number of features to select")

    return parser.parse_args()
