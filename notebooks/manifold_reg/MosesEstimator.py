__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
import os
import subprocess as subp
import tempfile as tmp
import pandas as pd
import numpy as np
import scipy
import re
from feature_parser import *

patt = re.compile(r"(-?\d+) (.+) \[(.+)\]")

class MosesEstimator(BaseEstimator):
    """
    A wrapper sklearn api based classifier that uses [MOSES](https://github.com/opencog/moses) under the hood
    """

    def __init__(self, scorers=None, alpha=1.0, assoc_mat=None, num_evals=10000,
                    complexity_ratio=3, fs_algo="random", fs_size=10, num_models=100, threshold=1e-3,
                    n_draws=30, ensemble=True, random_state=42):

        """
        Initialize Moses estimator
        :param scorers: The feature selection scorers to use - moses parameter
        :param alpha: Specific parameter for the weighting the background feature selection penalty - moses parameter
        :param assoc_mat: The network that represents feature relationships
        :param num_evals: Num of evaluation before stopping MOSES run - moses parameter
        :param complexity_ratio: the ratio of the score to complexity, to be used as a penalty, when ranking the metapopulation for fitness - moses parameter
        :param fs_algo: Feature selection algorithm to use - moses parameter
        :param fs_size: Num of features to select at a time - moses parameter
        :param num_models: Num of models to return from a MOSES run - moses parameter
        :param threshold: Improvement threshold for feature selection algorithms - moses parameter
        :param n_draws: Num of times to randomly draw features during feature selection step; only makes sense if fs_algo=random is being used - moses parameter
        :param ensemble: Whether to do majority voting on the model outputs or return average output
        :param random_state: Random seed value - moses parameter
        """
        self.scorers = scorers
        self.alpha = alpha
        self.assoc_mat = assoc_mat
        self.num_evals = num_evals
        self.complexity_ratio = complexity_ratio
        self.fs_algo = fs_algo
        self.fs_size = fs_size
        self.num_models = num_models
        self.threshold = threshold
        self.n_draws = n_draws
        self.ensemble = ensemble
        self.random_state = random_state


    def get_params(self, deep=True):
        return {"scorers": self.scorers, "alpha": self.alpha, "assoc_mat": self.assoc_mat,
                "num_evals": self.num_evals, "complexity_ratio": self.complexity_ratio, "fs_algo": self.fs_algo, "n_draws": self.n_draws,
                "ensemble": self.ensemble, "fs_size": self.fs_size, "num_models": self.num_models,
                "threshold": self.threshold, "random_state": self.random_state}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y, output_label="y", moses_params=None):

        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        data_df = pd.DataFrame(data)
        cols = [f"f{i}" for i in range(X.shape[1])]
        cols.append("y")
        data_df.columns = cols
        self.train_file_ = tmp.NamedTemporaryFile().name
        data_df.to_csv(self.train_file_, index=False)
        self.out_file_ = tmp.NamedTemporaryFile().name
        # print(self.out_file_)
        self.out_label_ = output_label
        return self._fit(moses_params)


    def predict(self, X):
        X_df = pd.DataFrame(X)
        X_df.columns = [f"f{i}" for i in range(X.shape[1])]
        res = self._eval_models(self.models_, X_df)
        # return res
        if self.ensemble:
            return (scipy.stats.mode(res, axis=0).mode)[0]

        return np.sum(res, axis=0) / res.shape[0]

    def score(self, X, y):
        y_pred = self.predict(X)

        return roc_auc_score(y, y_pred)

    def _fit(self, moses_params=None):

        opt = f"-i {self.train_file_} -o {self.out_file_} -u {self.out_label_} " \
              f"-m {self.num_evals} -W 1 --output-cscore 1 --output-deme-id 1 --result-count {self.num_models} " \
              f"--complexity-ratio={self.complexity_ratio} --enable-fs=1 --fs-target-size={self.fs_size} --fs-algo={self.fs_algo} --fs-assoc-mat={self.assoc_mat} " \
              f"--random-seed={self.random_state} --fs-alpha={self.alpha} --fs-n-draws={self.n_draws} --fs-threshold={self.threshold} "

        for i in self.scorers:
            opt += f"--fs-scorer={i} "

        if moses_params is not None:
            opt += f" {moses_params} "

        rcode, stdout, stderr = MosesEstimator.run_moses(opt)

        if rcode != 0:
            raise RuntimeError(f"MOSES ran into an error with return code {rcode}. Here is the stderr output:\n{stderr.decode('utf-8')}")

        self.models_ = MosesEstimator.parse_models(self.out_file_)
        # print(f"{len(self.models_)} parsed!")

        return self

    @staticmethod
    def run_moses(mose_opts):
        cmd = ["asmoses"]

        for opt in mose_opts.split():
            cmd.append(opt)

        process = subp.Popen(args=cmd, stdout=subp.PIPE, stderr=subp.PIPE)

        stdout, stderr = process.communicate()

        return process.returncode, stdout, stderr

    def _eval_models(self, models, test_df):
        """
        Evaluate a list of model objects against an input file
        :param: models: list of model objects
        :param input_file: the location of the input file
        :return: matrix:
        nxm matrix where n is the number of models and m is the number of samples. the matrix contains the predicted
        output of each model on the sample
        """

        test_tmp_file = tmp.NamedTemporaryFile().name
        model_outs = np.zeros((len(models), test_df.shape[0]))
        # print(test_tmp_file)

        test_df.to_csv(test_tmp_file, index=False)
        for i, moses_model in enumerate(models):
            # print(moses_model.model)
            temp_eval_file = tmp.NamedTemporaryFile().name
            # print(temp_eval_file)
            cmd = ['aseval-table', "-i", test_tmp_file, "-c", moses_model.model, "-o", temp_eval_file]
            process = subp.Popen(args=cmd, stdout=subp.PIPE)

            stdout, stderr = process.communicate()
            try:
                if process.returncode == 0:
                    y_pred = np.genfromtxt(temp_eval_file, skip_header=1)
                    model_outs[i] = y_pred
                else:
                    print("Error: The following error raised by eval-table %s" % stderr.decode("utf-8"))
                    raise ChildProcessError(stderr.decode("utf-8"))
            except ValueError as e:
                print(f"Error occured while evaluating: {moses_model.model}")
                print(f"Eval file - {temp_eval_file}")
                raise e

        return model_outs

    @staticmethod
    def _mae(scores, labels):
        return np.sum(np.abs(scores - labels)) / float(np.size(labels))

    @staticmethod
    def parse_models(combo_file):
        models = []
        fp = open(combo_file, "r")
        for line in fp:
            match = patt.match(line.strip())
            if match is not None:
                model = match.group(2).strip()
                # complexity = match.group(3).split(",")[2].split("=")[1]
                # incons_pen = match.group(3).split(",")[5].split("=")[1]
                models.append(MosesModel(model))

        return models



def feature_count(models, causal_feats):
    feat_count = {}
    for m in models:
        feats = m.get_features()
        for f in set(feats):
            if not f in feat_count:
                feat_count[f] = {"count": 1, "causal": "No"}
                # feat_count[f]["dist"] = np.sum(dist_mat[f,:np.array(causal_feats)], axis=1)
            else:
                feat_count[f]["count"] += 1

            if f in causal_feats:
                feat_count[f]["causal"] = "Yes"

    return feat_count

class MosesModel:
    """
    Specifies a parsed MOSES Combo model - used for analyzing the features in the model
    """
    def __init__(self, txt):
        self.model = txt
        self.features = None

    def get_features(self):
        if self.features is None:
            tree = combo_parser.parse(self.model)
            transformer = ComboTreeTransform()
            transformer.transform(tree)
            self.features = [int(f) for f in transformer.features]
            return self.features

        return self.features