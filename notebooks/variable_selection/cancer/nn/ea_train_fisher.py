#!/usr/bin/env python
# Author Abdulrahman S. Omar<hsamireh@gmail>
import jax.random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import time
import datetime
import argparse
import random
import operator
import itertools
from deap import tools, creator, base, gp
import ea_utils
from nn_util import setup_logger, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="Run Evolutionary Algorithm on selected feature sets.")

    parser.add_argument("--dataset", type=str, default=None, required=True, help="path to the dataset")
    parser.add_argument("--exp-dir", type=str, default=None, required=True, help="path to the experiment result dir")
    parser.add_argument("--dir", type=str, default=None, required=True, help="directory to save results")
    parser.add_argument("--seed", type=str, default=None, required=True, help="path to the seeds file")
    parser.add_argument("--num-feats", type=int, default=70, help="Number of features to select")
    parser.add_argument("--out-label", type=str, default="posOutcome", help="The column name of the output label")
    parser.add_argument("--cxpb", type=float, default=0.5, help="Cross-Over Probability")
    parser.add_argument("--mutpb", type=float, default=0.2, help="Mutation Probability")
    parser.add_argument("--num-gen", type=int, default=1000, help="Num of generations")

    return parser.parse_args()


def get_primitive_set(num_feats):
    def if_then_else(input, output1, output2):
        if input: return output1
        else: return output2

    def protectedDiv(left, right):
        try: return left / right
        except ZeroDivisionError: return 1

    # pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(bool, num_feats), bool, "f")
    #
    # pset.addPrimitive(operator.and_, [bool, bool], bool)
    # pset.addPrimitive(operator.or_, [bool, bool], bool)
    # pset.addPrimitive(operator.xor, [bool, bool], bool)
    # pset.addPrimitive(operator.not_, [bool], bool)
    # pset.addPrimitive(if_then_else, [bool, bool, bool], bool)

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, num_feats), bool, "f")

    # boolean operators
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)

    # floating point operators
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float ,float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)

    # logic operators
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    return pset

def fitness_fn(individual, x, y, pset):

    # Transform the tree expression into a callable function
    func = gp.compile(individual, pset)

    output = np.array([func(*x[i]) for i in range(len(x))]).astype(np.int32)
    # incorrect = np.sum((y_train != output).astype(np.float32))
    # return np.mean(accuracy),
    # return f1_score(y_train, output),
    return roc_auc_score(y, output), len(individual)

def get_ind_pred(ind, x, pset):
    # Get predicted output of an individual program
    func = gp.compile(ind, pset)

    output = np.array([func(*x[i]) for i in range(len(x))]).astype(np.int32)
    return output

def run_logistc_regression(X_train, X_val, X_test, y_train, y_val, y_test, cv, logger=None):

    log_param_grid = {"C": np.logspace(-2, 1, 10)}
    log_grid_cv = GridSearchCV(estimator=LogisticRegression(max_iter=10000), param_grid=log_param_grid, verbose=1,
                               scoring="roc_auc", cv=cv).fit(X_train, y_train)

    if logger is not None:
        logger.info(f"LR best params {log_grid_cv.best_params_}")
    clf = LogisticRegression(max_iter=10000,  **log_grid_cv.best_params_)
    # log_cv_score = np.mean(cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=cv))
    clf.fit(X_train, y_train)
    log_val_score = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
    log_test_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    if logger is not None:
        logger.info(f"LR scores - cv score: {log_val_score: .4f}, test_score: {log_test_score: .4f}")

    return clf, log_grid_cv.best_params_, log_val_score, log_test_score

def gen_early_stop_fn(x_val, y_val, pset, min_gen=100):

    def early_stop_fn(hof, n_gen, prev_fitness):
        val_preds = []
        for ind in hof:
            val_pred = get_ind_pred(ind, x_val, pset)
            val_preds.append(val_pred)

        val_preds = np.array(val_preds).T

        val_score = roc_auc_score(y_val, np.mean(np.array(val_preds), axis=1))

        stop = (n_gen > min_gen)  and (val_score < prev_fitness)
        return val_score, stop

    return early_stop_fn

def tree_similarity(ind1, ind2):
    # Function to measure the similarity of trees - we use fitness value to measure similairty
    fit_1, fit_2 = ind1.fitness.values[0], ind2.fitness.values[0]

    dist = np.abs(fit_1 - fit_2)

    return dist < 0.01


def run_deap(rng_key, X_train, X_val, X_test, y_train, y_val,
             y_test, cxpb, mutbp, cmpx_pen, num_gen, init_pop, logger=None):

    p = X_train.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0, -cmpx_pen))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    pset = get_primitive_set(p)

    # Conver the ndarray to tuple for passing to the individual programs
    x_train, x_val, x_test = tuple(map(tuple, X_train)), tuple(map(tuple, X_val)), \
                             tuple(map(tuple, X_test))

    toolbox = base.Toolbox()
    toolbox.register("expr", ea_utils.genHalfAndHalf, pset=pset, min_=1,  max_=3)
    toolbox.register("individual", ea_utils.initIterate, container=creator.Individual, generator=toolbox.expr)
    toolbox.register("population", ea_utils.initRepeat, container=list, func=toolbox.individual)

    toolbox.register("evaluate", fitness_fn, pset=pset, x=x_train, y=y_train)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", ea_utils.cxOnePoint)
    toolbox.register("expr_mut", ea_utils.genFull, min_=0, max_=3)
    toolbox.register("mutate", ea_utils.mutUniform, expr=toolbox.expr_mut, pset=pset)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    early_stop_fn = gen_early_stop_fn(x_val, y_val, pset)

    pop_keys = jax.random.split(rng_key, init_pop)

    pop = toolbox.population(keys=pop_keys, n=init_pop)
    hof = tools.ParetoFront(tree_similarity)
    _, logbook = ea_utils.eaSimple(rng_key ,pop, toolbox, cxpb, mutbp, num_gen, stats=mstats, halloffame=hof,
                                   early_stop_fn=early_stop_fn, verbose=False)

    train_preds = []
    val_preds = []
    test_preds = []
    for ind in hof:
        train_pred = get_ind_pred(ind, x_train, pset)
        train_preds.append(train_pred)

        val_pred = get_ind_pred(ind, x_val, pset)
        val_preds.append(val_pred)

        test_pred = get_ind_pred(ind, x_test, pset)
        test_preds.append(test_pred)

    train_preds = np.array(train_preds).T
    val_preds = np.array(val_preds).T
    test_preds = np.array(test_preds).T

    val_score = roc_auc_score(y_val, np.mean(np.array(val_preds), axis=1))
    test_score = roc_auc_score(y_test, np.mean(np.array(test_preds), axis=1))

    if logger is not None:
        logger.info(f"Num models: {len(hof)}, EA Validation Score: {val_score}, Test Score: {test_score}")

    return hof, val_score, test_score, train_preds, val_preds, test_preds


def run_seed(seed, X_df, y_df, exp_path, save_dir, num_feats, cxpb, mutpb, n_gen):

    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)
    start_time = time.time()
    logger = setup_logger(save_dir, seed)
    result_summary_dict = {"seed": [seed, seed], "classifier":
        ["BNN + DEAP", "BNN + DEAP + LR"], "num_feats": [], "top_5_feats": [] ,"cv_score": [], "test_score": []}

    cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

    idx_sig = np.load(f"{exp_path}/fisher_idx_sig_{seed}.npy")
    selected_idx = np.load(f"{exp_path}/bnn_sel_idx_s_{seed}_n_{num_feats}.npy")

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=seed, shuffle=True, stratify=y_df)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=seed, shuffle=True,
                                                        stratify=y_df, test_size=0.3)

    X_train_sig, X_test_sig = X_train.iloc[:, idx_sig], X_test.iloc[:,idx_sig]
    X_train_sel, X_test_sel = X_train_sig.iloc[:,selected_idx].to_numpy(), X_test_sig.iloc[:,selected_idx].to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    X_train_sel, X_val_sel, y_train, y_val = train_test_split(X_train_sel, y_train, random_state=seed, shuffle=True,
                                                              stratify=y_train, test_size=0.2)


    hof, val_score, test_score, train_preds, val_preds, test_preds = run_deap(rng_key, X_train_sel, X_val_sel, X_test_sel, y_train, y_val, y_test, cxpb, mutpb, n_gen, logger)

    X_train_sel_ea = np.concatenate([X_train_sel, train_preds], axis=1)
    X_val_sel_ea = np.concatenate([X_val_sel, val_preds], axis=1)
    X_test_sel_ea = np.concatenate([X_test_sel, test_preds], axis=1)

    clf_log, log_best_params, log_cv_score, log_test_score = run_logistc_regression(X_train_sel_ea, X_val_sel_ea, X_test_sel_ea,
                                                                            y_train, y_val ,y_test, cv, logger)

    # Check the top 5 feats acc. Logistic Regression classifier
    top_5_feats = ",".join([str(idx) for idx in np.argsort(np.abs(clf_log.coef_[0]))[::-1][:5]])

    result_summary_dict["cv_score"].append(val_score)
    result_summary_dict["test_score"].append(test_score)
    result_summary_dict["num_feats"].append(len(hof))
    result_summary_dict["top_5_feats"].append("-")

    result_summary_dict["cv_score"].append(log_cv_score)
    result_summary_dict["test_score"].append(log_test_score)
    result_summary_dict["num_feats"].append(num_feats + len(hof))
    result_summary_dict["top_5_feats"].append(top_5_feats)

    # Save everything
    result_summary_df = pd.DataFrame(result_summary_dict)
    result_summary_df.to_csv(f"{save_dir}/res_summary_fisher_genes_deap_s_{seed}.csv", index_label=False)
    pickle.dump(hof, open(f"{save_dir}/pareto_front_s_{seed}.pickle", "wb"))
    pickle.dump(log_best_params, open(f"{save_dir}/deap_lr_best_params_s_{seed}.pickle", "wb"))

    np.save(f"{save_dir}/deap_train_eval_out_s_{seed}.npy", train_preds)
    np.save(f"{save_dir}/deap_val_eval_out_s_{seed}.npy", test_preds)
    np.save(f"{save_dir}/deap_test_eval_out_s_{seed}.npy", test_preds)

    end_time = time.time()

    elapsed = datetime.timedelta(seconds=(end_time - start_time))
    logger.info(f"Done for seed {seed}. Time elapsed - {elapsed}")


def main():

    args = parse_args()
    dataset_path = args.dataset
    save_dir_path = args.dir
    exp_dir_path = args.exp_dir
    seed_path = args.seed
    nfeats = args.num_feats
    out_col = args.out_label

    cxpb = args.cxpb
    mutpb = args.mutpb
    num_gen = args.num_gen


    df = pd.read_csv(dataset_path)
    X_df, y_df = df[df.columns.difference([out_col])], df[out_col]

    exp_seeds = []
    with open(seed_path, "r") as fp:
        for line in fp.readlines():
            exp_seeds.append(int(line.strip()))

    print(f"Total {len(exp_seeds)} runs")

    for seed in exp_seeds:
        print(f"Running seed {seed}")
        try:
            run_seed(seed, X_df, y_df, exp_dir_path, save_dir_path, nfeats, cxpb, mutpb, num_gen)

        except Exception as e:
            print(f"Ran into an error {e} while running seed {seed}. Skipping it..")

    print("Done!")

if __name__ == "__main__":
    main()