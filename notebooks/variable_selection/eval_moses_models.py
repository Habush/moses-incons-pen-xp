#!/home/xabush/moses-incons-pen-xp/venv/bin/python3
__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import os
import subprocess
import re
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import argparse
import tempfile
import glob
import scipy

patt = re.compile(r"(-?\d+) (.+) \[(.+)\]")


def parse_args():
    parser = argparse.ArgumentParser(prog="parse-eval-models", description="Parse Moses models and run eval-table to evaluate their scores")
    parser.add_argument("--seed", type=str, default=None, help="path to the seeds file")
    parser.add_argument("-t", "--target", help="target feature", default="posOutcome")
    parser.add_argument("--dir", type=str, default=None, help="path to the directory containing moses runs with different random seeds")
    parser.add_argument("--test", type=str, default=None, help="path to test file")
    parser.add_argument("--combo", type=str, default=None, help="path to combo file")

    args = parser.parse_args()

    return args

def parse_models(combo_file):
    models = []
    with open(combo_file, "r") as fp:
        for line in fp:
            match = patt.match(line.strip())
            if match is not None:
                model = match.group(2).strip()
                models.append(model)

    return models

def eval_models(models, test_file, target, seed=42):
    """
    Evaluate a list of model objects against an input file
    :param: models: list of model objects
    :param input_file: the location of the input file
    :return: matrix:
    nxm matrix where n is the number of models and m is the number of samples. the matrix contains the predicted
    output of each model on the sample
    """

    input_df = pd.read_csv(test_file)
    y_true = input_df[target].to_numpy()
    temp_eval_file = tempfile.NamedTemporaryFile().name
    eval_log = tempfile.NamedTemporaryFile().name
    model_outs = np.zeros((len(models), y_true.shape[0]))
    for i, moses_model in enumerate(models):
        cmd = ['eval-table', "-i", test_file, "-c", moses_model, "-o", temp_eval_file, "-u",
               target, "-f", eval_log, "-r", str(seed)]
        process = subprocess.Popen(args=cmd, stdout=subprocess.PIPE)

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            y_pred = np.genfromtxt(temp_eval_file, skip_header=1)
            model_outs[i] = y_pred
        else:
            print("Error: The following error raised by eval-table %s" % stderr.decode("utf-8"))
            raise ChildProcessError(stderr.decode("utf-8"))


    return model_outs, y_true

def mae(scores, labels):
  return np.sum(np.abs(scores - labels)) / float(np.size(labels))

def score_models(models_out, y_true, ensemble=True):
    # models out has shape mxn - m models, n samples
    if ensemble: #do majority voting for each sample
        y_vote = scipy.stats.mode(models_out, axis=0).mode
        # y_vote = y_vote.reshape(-1, 1) # this was the source of the bug in scoring as np broadcasted the second array
        y_vote = np.ravel(y_vote)
        assert y_vote.shape[0] == y_true.shape[0]
        # clf = LogisticRegression()
        # clf.fit(y_vote, y_true)
        # y_pred = clf.predict_proba(y_vote)
        return mae(y_vote, y_true)

    else:
        total_loss = 0.0
        for i in range(models_out.shape[0]):
            y_model = models_out[i]
            # y_model = y_model.reshape(-1, 1)
            y_model = np.ravel(y_model)
            assert y_model.shape[0] == y_true.shape[0]
            # clf = LogisticRegression()
            # clf.fit(y_model, y_true)
            # y_pred = clf.predict_proba(y_model)
            total_loss += mae(y_model, y_true)

        return total_loss / models_out.shape[0]

def models_to_df(models, seed):

    data = {"model": [], "complexity": [], "inconsistency_pen": [], "fold": [], "seed": [], "mse_train": [],
            "recall_train": [], "precision_train": [], "balanced_acc_train": [], "f1_train": [], "mse_test": [],
            "recall_test": [], "precision_test": [], "balanced_acc_test": [], "f1_test": []
            }

    for model in models:
        for k in data:
            if k == "seed":
                data[k].append(seed)
            else:
                data[k].append(model[k])

    res_df = pd.DataFrame(data)

    return res_df


def score_moses_models(dir, seeds, test=None, combo_file=None, ensemble=True, target="out"):

    if test is None:
        if dir is None or not os.path.exists(dir):
            print(f"Error: working directory {dir} doesn't exist or cannot be found! Make sure it exists and writable")
            exit(-1)
        folds = []
        fold_pat = re.compile("fold_(\d+)_val.csv")
        for file in os.listdir(dir):
            if fold_pat.match(file):
                folds.append(os.path.join(dir, file))
        for seed in seeds:
            seed_path = os.path.join(dir, "seed_{0}".format(seed))
            print(f"Evaluating seed {seed}")
            fold_dict = {"fold":[], "mae": []}
            for i, fold in enumerate(folds):
                match = fold_pat.match(os.path.basename(fold))
                i = match.group(1)
                combo_file = f"combo_s_{seed}_f_{i}.txt"
                combo_file = os.path.join(seed_path, combo_file)
                fold_models = parse_models(combo_file)
                model_out, y_true = eval_models(fold_models, fold, target, seed)
                np.save(os.path.join(seed_path, f"model_out_f_{i}.npy"), model_out)
                score = score_models(model_out, y_true, ensemble)
                fold_dict["fold"].append(i)
                fold_dict["mae"].append(score)

            fold_score_df = pd.DataFrame(fold_dict)
            return fold_score_df

    elif test is not None and combo_file is not None:
        models = parse_models(combo_file)
        model_out, y_true = eval_models(models, test, target)
        score = score_models(model_out, y_true, ensemble)
        return score
