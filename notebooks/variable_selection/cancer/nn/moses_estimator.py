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

patt = re.compile(r"([+-]?[0-9]*[.]?[0-9]+) (.+) \[(.+)\]")


### Needed b/c moses gets confused by when column names are numbers
def assign_cols(X):
    X_copy = X.copy()
    orig_cols = X.columns.to_list()
    cols = []
    n_cols = len(orig_cols)
    for i in range(n_cols):
        cols.append(f"f{orig_cols[i]}")
    X_copy.columns = cols

    return X_copy

class MosesEstimator(BaseEstimator):
    """
    A wrapper sklearn api based classifier that uses [MOSES](https://github.com/opencog/moses) under the hood
    """

    def __init__(self, seed, num_evals=10000, complexity_ratio=10, num_models=100, prob="it",
                 complexity_temp=2000, hc_fraction_nn=0.1, hc_crossover_min_nn=500, hc_crossover_pop_size=100,
                 diversity_pressure=0.5):

        """
        Initialize Moses estimator
        :param seed: Random seed value - moses parameter
        :param num_evals: Num of evaluation before stopping MOSES run - moses parameter
        :param complexity_ratio: the ratio of the score to complexity, to be used as a penalty, when ranking the metapopulation for fitness - moses parameter
        :param num_models: Num of models to return from a MOSES run - moses parameter
        :param prob: What score is MOSES maximizing - moses parameter
        """
        self.seed = seed
        self.num_evals = num_evals
        self.complexity_ratio = complexity_ratio
        self.prob = prob
        self.num_models = num_models
        self.complexity_temp = complexity_temp
        self.hc_fraction_nn = hc_fraction_nn
        self.hc_crossover_min_nn = hc_crossover_min_nn
        self.hc_crossover_pop_size = hc_crossover_pop_size
        self.diversity_pressure = diversity_pressure

        self.tmp_folder = tmp.TemporaryDirectory()
        self.tmp_dir_name = self.tmp_folder.name

    def get_params(self, deep=True):
        return {"num_evals": self.num_evals, "complexity_ratio": self.complexity_ratio, "num_models": self.num_models,
                "seed": self.seed, "prob": self.prob, "complexity_temp": self.complexity_temp,
                "hc_fraction_nn": self.hc_fraction_nn, "hc_crossover_min_nn": self.hc_crossover_min_nn,
                "hc_crossover_pop_size": self.hc_crossover_pop_size, "diversity_pressure": self.diversity_pressure}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y, output_label="y", moses_params=None):
        # data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        data_df = pd.concat([assign_cols(X), y], axis=1)
        self.train_file_ = os.path.join(self.tmp_dir_name, "train_file.csv")
        # print(self.train_file_)
        data_df.to_csv(self.train_file_, index=False)
        self.out_file_ = os.path.join(self.tmp_dir_name, "out_file.txt")
        self.log_file_ = os.path.join(self.tmp_dir_name, "log_file.txt")
        self.out_label_ = output_label
        return self._fit(moses_params)


    def predict(self, X):
        res = self._eval_models(self.models_, assign_cols(X))
        return np.mean(res, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)

        return roc_auc_score(y, y_pred)


    def _fit(self, moses_params=None):
        # print(log_file)

        opt = ["-i", self.train_file_, "-o", self.out_file_, "-u", self.out_label_, "-m", f"{self.num_evals}", "-W", "1",
               "--output-cscore", "1", "--output-deme-id", "1", "--result-count", f"{self.num_models}", "--complexity-ratio",
               f"{self.complexity_ratio}", "--random-seed", f"{self.seed}", "-H", self.prob,
               "--complexity-temperature", f"{self.complexity_temp}", "--hc-crossover-min-neighbors", f"{self.hc_crossover_min_nn}", "--hc-crossover-pop-size",
               f"{self.hc_crossover_pop_size}", "--hc-fraction-of-nn", f"{self.hc_fraction_nn}", "--diversity-autoscale", "1",
               "--diversity-pressure", f"{self.diversity_pressure}",
               "-f", self.log_file_]

        if moses_params is not None:
            for param in moses_params:
                opt.append(param)

        rcode, stdout, stderr = MosesEstimator.run_moses(opt)

        if rcode != 0:
            raise RuntimeError(f"MOSES ran into an error with return code {rcode}. Here is the stderr output:\n{stderr.decode('utf-8')}")

        self.models_ = MosesEstimator.parse_models(self.out_file_)
        # print(f"Model output file {self.out_file_}")
        # print(f"{len(self.models_)} parsed!")

        return self

    @staticmethod
    def run_moses(mose_opts):
        # cmd = ["asmoses"]

        # for opt in mose_opts.split():
        #     cmd.append(opt)
        mose_opts = ["asmoses"] + mose_opts
        # print(mose_opts)
        process = subp.Popen(mose_opts, stdout=subp.PIPE, stderr=subp.PIPE)

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

        test_tmp_file = os.path.join(self.tmp_dir_name, "test_tmp_file")
        model_outs = np.zeros((len(models), test_df.shape[0]))
        # print(model_outs.shape)
        test_df.to_csv(test_tmp_file, index=False)
        # print(test_tmp_file)
        temp_eval_file = os.path.join(self.tmp_dir_name, "tmp_eval_file")
        for i, moses_model in enumerate(models):
            # print(moses_model.model)

            # print(temp_eval_file)
            cmd = ['aseval-table', "-i", test_tmp_file, "-c", moses_model.model, "-o", temp_eval_file,
                   "-L", "1"]

            process = subp.Popen(args=cmd, stdout=subp.PIPE)

            stdout, stderr = process.communicate()
            try:
                if process.returncode == 0:
                    y_pred = []
                    with open(temp_eval_file, "r") as fp:
                        for k, line in enumerate(fp.readlines()):
                            if k == 0 or line.strip() == "":
                                continue
                            y_pred.append(int(line.strip()))

                    # print(len(y_pred))
                    # y_pred = np.genfromtxt(temp_eval_file, skip_header=1)
                    model_outs[i] = np.array(y_pred, dtype=np.int)
                else:
                    print("Error: The following error raised by eval-table %s" % stderr.decode("utf-8"))
                    raise ChildProcessError(stderr.decode("utf-8"))
            except ValueError as e:
                print(f"Error occured while evaluating: {moses_model.model}")
                print(f"Eval file - {temp_eval_file}, Test file - {test_tmp_file}")
                raise e

        return model_outs

    def cleanup(self):
        self.tmp_folder.cleanup()

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
                if model == "true" or model == "false":
                    continue
                # complexity = match.group(3).split(",")[2].split("=")[1]
                # incons_pen = match.group(3).split(",")[5].split("=")[1]
                models.append(MosesModel(model))

        return models


def feature_count(models, causal_feats, assoc_mat):
    feat_count = {}
    for m in models:
        feats = m.get_features()
        for f in feats:
            if not f in feat_count:
                feat_count[f] = {"count": 1, "causal": "No"}
                feat_count[f]["dist"] = np.sum(assoc_mat[f-1][np.array(causal_feats)])
            else:
                feat_count[f]["count"] += 1

            if f in causal_feats:
                feat_count[f]["causal"] = "Yes"

    return feat_count

def feature_count(models):
    feat_count = {}
    for m in models:
        feats = m.get_features()
        for f in feats:
            if not f in feat_count:
                feat_count[f] = 1
            else:
                feat_count[f] += 1

    return {k : v for k, v in sorted(feat_count.items(), key=lambda i: i[1], reverse=True)}

class MosesModel:
    """
    Specifies a parsed MOSES Combo model - used for analyzing the features in the model
    """
    def __init__(self, txt):
        self.model = txt

    def get_features(self):
        tree = combo_parser.parse(self.model)
        transformer = ComboTreeTransform()
        transformer.transform(tree)
        features = [int(f) - 1 for f in transformer.features]
        return features