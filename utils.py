__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import pandas as pd
from moses_cross_val.main import moses_runner, model_evaluator
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, auc, f1_score
from mlxtend.evaluate import scoring
from moses_cross_val.models.objmodel import Score, MosesModel

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

def parse_combo_models(combo_file, train_file_path,
                       test_file_path, target_feature="posOutcome"):
    runner = moses_runner.MosesRunner(train_file_path, "",
                                      "", target_feature="posOutcome")
    combo_models = runner.format_combo(combo_file)
    print("Num models: ", len(combo_models))

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

    custom_models = [CustomModel(m) for m in combo_models]

    data = {"model": [], "complexity": [], "inconsistency_pen": [],
            "recall_train": [], "precision_train": [], "balanced_acc_train": [], "f1_train": [], "spec_train": [],
            "recall_test": [], "precision_test": [], "balanced_acc_test": [], "f1_test": [], "spec_test": [],
            }

    for model in custom_models:
        for k in data:
            data[k].append(model[k])

    res_df = pd.DataFrame(data)

    return res_df