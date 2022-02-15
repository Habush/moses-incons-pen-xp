#!/home/xabush/moses-incons-pen-xp/venv/bin/python3
__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import os
import subprocess
import re
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import argparse
import tempfile
import glob

patt = re.compile(r"(-?\d+) (.+) \[(.+)\]")

class Model:

    def __init__(self, model, complexity, incons_pen):
        self.model = model
        self.complexity = complexity
        self.inconsistency_pen = incons_pen

        self.mse_train = None
        self.recall_train = None
        self.precision_train = None
        self.balanced_acc_train = None
        self.f1_train = None

        self.mse_test = None
        self.recall_test = None
        self.precision_test = None
        self.balanced_acc_test = None
        self.f1_test = None

        self.fold = None,

    def __dict__(self):
        return {"model": self.model, "complexity": self.complexity, "inconsistency_pen": self.inconsistency_pen,
                "fold": self.fold,
                "mse_test": self.mse_test,
                "recall_test": self.recall_test,
                "precision_test": self.precision_test,
                "balanced_acc_test": self.balanced_acc_test,
                "f1_test": self.f1_test,
                "mse_train": self.mse_train,
                "recall_train": self.recall_train,
                "precision_train": self.precision_train,
                "balanced_acc_train": self.balanced_acc_train,
                "f1_train": self.f1_train}

    def __getitem__(self, item):
        return self.__dict__()[item]

def parse_args():
    parser = argparse.ArgumentParser(prog="parse-eval-models", description="Parse Moses models and run eval-table to evaluate their scores")
    parser.add_argument("test", help="path to test file")
    parser.add_argument("dir", help="path to the directory containing moses runs with different random seeds")
    parser.add_argument("seed", help="path to the seeds file")
    parser.add_argument("-t", "--target", help="target feature", default="posOutcome")

    args = parser.parse_args()

    return args

def parse_models(combo_file):
    models = []
    with open(combo_file, "r") as fp:
        for line in fp:
            match = patt.match(line.strip())
            if match is not None:
                model = match.group(2).strip()
                if model == "true" or model == "false":
                    continue
                complexity = match.group(3).split(",")[2].split("=")[1]
                # incons_pen = match.group(3).split(",")[5].split("=")[1]
                models.append(Model(model, complexity, ""))

    return models

def eval_models(models, test_file, target, seed, train_set=False):
    """
    Evaluate a list of model objects against an input file
    :param: models: list of model objects
    :param input_file: the location of the input file
    :return: matrix:
    nxm matrix where n is the number of models and m is the number of samples. the matrix contains the predicted
    output of each model on the sample
    """

    input_df = pd.read_csv(test_file)
    y_true = input_df[target].to_numpy(dtype=int)

    temp_eval_file = tempfile.NamedTemporaryFile().name
    eval_log = tempfile.NamedTemporaryFile().name

    for i, moses_model in enumerate(models):
        cmd = ['eval-table', "-i", test_file, "-c", moses_model.model, "-o", temp_eval_file, "-u",
               target, "-f", eval_log, "-r", str(seed)]
        process = subprocess.Popen(args=cmd, stdout=subprocess.PIPE)

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            y_pred = np.genfromtxt(temp_eval_file, skip_header=1, dtype=int)
            balanced_acc, prec = balanced_accuracy_score(y_true, y_pred), precision_score(y_true, y_pred)
            recall, f1 = recall_score(y_true, y_pred), f1_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            if train_set:
                moses_model.mse_train = mse
                moses_model.recall_train = recall
                moses_model.precision_train = prec
                moses_model.balanced_acc_train = balanced_acc
                moses_model.f1_train = f1
            else:
                moses_model.mse_test = mse
                moses_model.recall_test = recall
                moses_model.precision_test = prec
                moses_model.balanced_acc_test = balanced_acc
                moses_model.f1_test = f1

        else:
            print("Error: The following error raised by eval-table %s" % stderr.decode("utf-8"))
            raise ChildProcessError(stderr.decode("utf-8"))

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


def main(test, dir, seeds, target="posOutcome"):

    folds = []
    fold_pat = re.compile("fold_(\d+)_val.csv")
    for file in os.listdir(dir):
        if fold_pat.match(file):
            folds.append(os.path.join(dir, file))

    for seed in seeds:
        seed_path = os.path.join(dir, "seed_{0}".format(seed))
        models = []
        print(f"Evaluating seed {seed}")
        for fold in folds:
            match = fold_pat.match(os.path.basename(fold))
            i = match.group(1)
            combo_file = f"combo_s_{seed}_f_{i}.txt"
            combo_file = os.path.join(seed_path, combo_file)
            fold_models = parse_models(combo_file)
            eval_models(fold_models, fold, target, seed, True)
            eval_models(fold_models, test, target, seed, False)
            for model in fold_models:
               model.fold = i

            models.extend(fold_models)

        df = models_to_df(models, seed)

        df.to_csv(os.path.join(seed_path, "combo_models.csv"), index=False)

    print("Done!")


if __name__ == "__main__":

    args = parse_args()

    # make sure the provided files actually exists
    if not os.path.exists(args.test):
        print(f"Error: train file {args.test} doesn't exist or cannot be found!")
        exit(-1)

    if not os.path.exists(args.dir):
        print(f"Error: working directory {args.dir} doesn't exist or cannot be found! Make sure it exists and writable")
        exit(-1)

    if not os.path.exists(args.seed):
        print(f"Error: seed file {args.seed} doesn't exist or cannot be found!")
        exit(-1)

    seeds = []

    with open(args.seed, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    main(args.test, args.dir, seeds, args.target)