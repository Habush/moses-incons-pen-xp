__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import pandas as pd
from moses_cross_val.main import moses_runner, model_evaluator
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, auc, f1_score
from mlxtend.evaluate import scoring
from moses_cross_val.models.objmodel import Score, MosesModel
import os
import re
import scipy.stats

moses_options = "-j 16 --balance 1 -m 100000 -W 1 --output-cscore 1 --result-count 100 " \
                "--reduct-knob-building-effort 2 --complexity-ratio 3 " \
                "--hc-crossover-min-neighbors 5000 --hc-fraction-of-nn .4 " \
                "--hc-crossover-pop-size 1200 --diversity-autoscale 1 " \
                "-f asmoses-top-100-feats.log -l DEBUG "

moses_options_comp_6 = "-j 16 --balance 1 -m 100000 -W 1 --output-cscore 1 --result-count 100 " \
                "--reduct-knob-building-effort 2 --complexity-ratio 3 " \
                "--hc-crossover-min-neighbors 5000 --hc-fraction-of-nn .4 " \
                "--hc-crossover-pop-size 1200 --diversity-autoscale 1 " \
                "-f asmoses-top-100-feats.log -l DEBUG "

moses_options_w_div = moses_options + " --diversity-pressure 0.8  "
moses_options_w_pen = moses_options + " --rel-types IntensionalSimilarity --feature-type ConceptNode  --scm-path /home/xabush/moses-incons-pen-xp/data/intensional_sim_top_89.scm  "
moses_options_w_pen_div = moses_options + " --diversity-pressure 0.8 --rel-types IntensionalSimilarity --feature-type ConceptNode  --scm-path /home/xabush/moses-incons-pen-xp/data/intensional_sim_top_89.scm "

#%%

cross_val_opts = {"folds": 3, "testSize": 0.3, "randomSeed": 3}

class CustomModel:
    def __init__(self, model):
        self.model = model.model
        self.complexity = model.complexity
        self.inconsistency_pen = model.inconsistency_pen

        self.recall_train = model.train_score.recall
        self.precision_train = model.train_score.precision
        self.balanced_acc_train = model.train_score.accuracy
        self.f1_train = model.train_score.f1_score
        self.spec_train = model.train_score.p_value

        self.recall_test = model.test_score.recall
        self.precision_test = model.test_score.precision
        self.balanced_acc_test = model.test_score.accuracy
        self.f1_test = model.test_score.f1_score
        self.spec_test = model.test_score.p_value

    def __dict__(self):
        return {"model": self.model, "complexity": self.complexity, "inconsistency_pen": self.inconsistency_pen,
                "recall_test": self.recall_test,
                "precision_test": self.precision_test,
                "balanced_acc_test": self.balanced_acc_test,
                "f1_test": self.f1_test,
                "spec_test": self.spec_test,
                "recall_train": self.recall_train,
                "precision_train": self.precision_train,
                "balanced_acc_train": self.balanced_acc_train,
                "f1_train": self.f1_train,
                "spec_train": self.spec_train}

    def __getitem__(self, item):
        return self.__dict__()[item]


def load_features(path):
    feats = []
    with open(path, "r") as fp:
        for line in fp.readlines():
            feats.append(line.strip())

    return feats

def display_df_info(df):
    print("Dataset dims", df.shape)
    display(df.head())

def convert_dataframe_to_atomese(df, output_file, conf=1.0, link_type="IntensionalSimilarity",
                                 node1_type="GeneNode", node2_type="GeneNode"):
    """
    Converts a dataframe into an atomese representation. For example if the data is an intensional similiarty matrix
    b/n genes, it will convert it to the f.f format
    (IntensionalSimilarity (stv 0.02 1)
        (GeneNode "A")
        (GeneNode "B"))

    :param df: The pandas dataframe
    :param output_file: path to save the output file
    :param conf: Set the confidence value
    :param link_type: The link type to create b/n nodes
    :param node1_type: Type of the first outgoing node
    :param node2_type: Type of the second outgoing node
    :return:
    """
    template = "({0} (stv {1} {2})\n " \
               "    ({3} \"{4}\")\n" \
               "    ({5} \"{6}\"))\n"

    cols = df.columns.to_list()

    with open(output_file, "w") as fp:
        for col1 in cols:
            for col2 in cols:
                if col1 != col2:
                    strength = df[col1][col2]
                    link = template.format(link_type, strength, conf,
                                           node1_type, col1,
                                           node2_type, col2)
                    fp.write(link)

    print("Results saved to " + output_file)


def evaluate_combo_models(combo_models, train_file_path, test_file_path, target_feature="posOutcome"):

    moses_eval = model_evaluator.ModelEvaluator(target_feature)
    score_mat_train = moses_eval.run_eval(combo_models, train_file_path)
    score_mat_test = moses_eval.run_eval(combo_models, test_file_path)

    df_train, df_test = pd.read_csv(train_file_path), pd.read_csv(test_file_path)
    train_target, test_target = df_train[target_feature], df_test[target_feature]

    for row, model in zip(score_mat_train, combo_models):
        balanced_acc_train = balanced_accuracy_score(train_target, row)
        prec_train = precision_score(train_target, row)
        recall_train = recall_score(train_target, row)
        f1_train = f1_score(train_target, row)
        spec_train = scoring(train_target, row, metric="specificity")
        model.train_score = Score(recall_train, prec_train, balanced_acc_train, f1_train, spec_train)

    for row, model in zip(score_mat_test, combo_models):
        balanced_acc_test = balanced_accuracy_score(test_target, row)
        prec_test = precision_score(test_target, row)
        recall_test = recall_score(test_target, row)
        f1_test = f1_score(test_target, row)
        spec_test = scoring(test_target, row, metric="specificity")
        model.test_score = Score(recall_test, prec_test, balanced_acc_test, f1_test, spec_test)

    return [CustomModel(m) for m in combo_models]


def parse_combo_models(combo_file, train_file_path,
                       test_file_path, target_feature="posOutcome"):

    runner = moses_runner.MosesRunner(train_file_path, "",
                                      "", target_feature="posOutcome")
    combo_models = runner.format_combo(combo_file)
    print("Num models: ", len(combo_models))

    custom_models = evaluate_combo_models(combo_models, train_file_path, test_file_path)

    data = {"model": [], "complexity": [], "inconsistency_pen": [],
            "recall_train": [], "precision_train": [], "balanced_acc_train": [], "f1_train": [], "spec_train": [],
            "recall_test": [], "precision_test": [], "balanced_acc_test": [], "f1_test": [], "spec_test": [],
            }

    for model in custom_models:
        for k in data:
            data[k].append(model[k])

    res_df = pd.DataFrame(data)

    return res_df


def parse_combo_dir(dir_path, train_file_path, test_file_path, target_feature="posOutcome"):

    files = [f for f in os.listdir(dir_path) if re.match(r'fold_[0-9]+\.csv', f)]

    folds = {}
    for file in files:
        match = re.match("(fold_)([0-9]+)(\.csv)", file)
        fold = match.group(2)
        df = pd.read_csv(os.path.join(dir_path, file))
        fold_models, comps = df["model"].to_list(), df["complexity"].to_list()

        models = []

        for model, comp in zip(fold_models, comps):
            models.append(MosesModel(model, comp, -1))

        folds[fold] = evaluate_combo_models(models, train_file_path, test_file_path, target_feature)

    data = {"model": [], "complexity": [], "fold": [],
            "recall_train": [], "precision_train": [], "balanced_acc_train": [], "f1_train": [], "spec_train": [],
            "recall_test": [], "precision_test": [], "balanced_acc_test": [], "f1_test": [], "spec_test": [],
            }

    for fold in folds:
        for model in folds[fold]:
            for k in data:
                if k != "fold":
                    data[k].append(model[k])
                else:
                    data[k].append(fold)

    res_df = pd.DataFrame(data)

    return res_df

import inspect

def print_var(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    print(str([k for k, v in callers_local_vars if v is var][0]))

def do_stat_test(exp1, exp2, score, name_1="exp1", name_2="exp2", alpha=0.05):
    if score != "complexity":
        scores = [f"{score}_train", f"{score}_test"]
    else:
        scores = ["complexity"]
    comps = ["two-sided", "less", "greater"]
    dfs = []

    for sc in scores:
        print(f"\n=========================== {sc} ===============================\n")
        for comp in comps:
            df, sig_res, total = do_stat_test_helper(exp1, exp2, sc, alpha, alt=comp)

            if comp == "two-sided":
                print(f"Two tailed test - {sig_res}/{total} [{(sig_res / total) * 100}%] results are significant, (p_value < {alpha})")

            elif comp == "less":
                print(f"One tailed test - {name_1} < {name_2} - {sig_res}/{total} [{(sig_res / total) * 100}%] results are significant, (p_value < {alpha})")
            elif comp == "greater":
                print(f"One tailed test - {name_1} > {name_2} - {sig_res}/{total} [{(sig_res / total) * 100}%] results are significant, (p_value < {alpha})")

            dfs.append(df)

    return dfs


def do_stat_test_helper(exp1, exp2, score, alpha, num_folds = 4, alt="two-sided"):
    df_dict = {"seed": [], "fold": [], "normal": [], "test_statistic": [], "p_value": []}

    for s1, s2 in zip(exp1, exp2):
        df1 = exp1[s1]
        df2 = exp2[s2]

        for f in range(num_folds):

            df1_fold = df1[df1["fold"] == f][score]
            df2_fold = df2[df2["fold"] == f][score]

            if df1_fold.shape[0] != df2_fold.shape[0]:
                if df1_fold.shape[0] < df2_fold.shape[0]:
                    df2_fold = df2_fold.head(df1_fold.shape[0])
                else:
                    df1_fold = df1_fold.head(df2_fold.shape[0])

            # Null hypothesis in normality test is the samples are not from a normal distribution
            _, nt1 = scipy.stats.normaltest(df1_fold)
            _, nt2 = scipy.stats.normaltest(df2_fold)

            df_dict["seed"].append(s1)
            df_dict["fold"].append(f)

            if nt1 < alpha and nt2 < alpha:  # null hypothesis cannot be rejected - normal
                df_dict["normal"].append("Yes")
                t, p = scipy.stats.ttest_ind(df1_fold, df2_fold, alternative=alt)
                df_dict["test_statistic"].append(t)
                df_dict["p_value"].append(p)
            else:  # not normal - use nonparametric test
                df_dict["normal"].append("No")
                t, p = scipy.stats.wilcoxon(df1_fold, df2_fold, alternative=alt)
                df_dict["test_statistic"].append(t)
                df_dict["p_value"].append(p)

    df = pd.DataFrame(df_dict)
    sig_res = len(df[df["p_value"] < alpha])

    total = len(exp1) * num_folds

    return df, sig_res, total