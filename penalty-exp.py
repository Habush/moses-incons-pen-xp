#!/home/xabush/moses-incons-pen-xp/venv/bin/python3
__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

from sklearn.model_selection import StratifiedKFold, KFold
import os
import argparse
import subprocess
import pandas as pd
import datetime
import logging
import time
import sys

k_fold_seed = 42

moses_options = "-j 20 -m 20000 -W 1 --output-cscore 1 --output-deme-id 1 --result-count 100 " \
                "--reduct-knob-building-effort 2 " \
                "--hc-crossover-min-neighbors 5000 --hc-fraction-of-nn .4 " \
                "--hc-crossover-pop-size 1200 " \
                "-l DEBUG --output-format=combo"


def set_up_logging(wk_dir):
    logger = logging.getLogger("penalty-exp")
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(wk_dir, "app.log")

    # Create handlers
    c_handler = logging.StreamHandler(stream=sys.stdout)
    f_handler = logging.FileHandler(log_path)
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(prog="pen-exp", description="Inconsistency penalty MOSES experiment runner")
    parser.add_argument("train", help="path to the train file to be used as input")
    parser.add_argument("dir", help="directory to save results in")
    parser.add_argument("seed", help="path to seed.txt file that contains the random seeds")
    parser.add_argument("--mat", help="path to association matrix")
    parser.add_argument("--cpath", help="path to the continuous data")
    parser.add_argument("-s", "--scm_path", help="path of the scheme file containing the background"
                                                 "Atomspace. This is required if inconsistency penalty"
                                                 "is set ON")
    parser.add_argument("-p", "--inconsistency_pen", help="whether to apply inconsistency"
                                                          "penalty or not",
                        default=0, type=int, choices=[0, 1])


    parser.add_argument("-a", "--alpha", help="regularizer value for the network penalty",
                        default=0, type=float)
    parser.add_argument("-g", "--gamma", help="The gamma parameter for the Mahalanobis Kernel",
                        default=1, type=float)

    parser.add_argument("-c", "--complexity_ratio", help="complexity ratio value to use",
                        default=3, type=int)

    parser.add_argument("-d", "--diversity_pressure", help="Specifies value for diversity pressure",
                        default=0.0, type=float)

    parser.add_argument("-t", "--target", help="target feature", default="posOutcome")

    parser.add_argument("-f", "--fold", help="number of cross-validation folds", default=5, type=int)
    parser.add_argument("--fs", help="Enable feature selection [default = 0]", default=0, type=int, choices=[0, 1])
    parser.add_argument("-x", "--xopts", help="Extra options for MOSES", default=None, type=str)

    args = parser.parse_args()

    return args


def split_dataset(train_file, n_folds, wk_dir, contin_path, target):
    df = pd.read_csv(train_file)
    df_contin = pd.read_csv(contin_path)
    X, y = df[df.columns.difference([target])], df[target]
    X_contin, y_contin = df_contin[df_contin.columns.difference([target])], df_contin[target]
    if n_folds > 0:
        skf = StratifiedKFold(n_splits=n_folds, random_state=k_fold_seed, shuffle=True)

        fold = 0

        folds = {}

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_train_c, y_train_c = X_contin.iloc[train_idx], y_contin.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            df_train = pd.concat([y_train, X_train], axis=1)
            df_train_c = pd.concat([y_train_c, X_train_c], axis=1)
            df_val = pd.concat([y_val, X_val], axis=1)
            df_train_path = os.path.join(wk_dir, f"fold_{fold}_train.csv")
            df_val_path = os.path.join(wk_dir, f"fold_{fold}_val.csv")
            df_train_c_path = os.path.join(wk_dir, f"fold_{fold}_train_contin.csv")

            df_train.to_csv(df_train_path, index=False)
            df_val.to_csv(df_val_path, index=False)
            df_train_c.to_csv(df_train_c_path, index=False)

            folds[fold] = [df_train_path, df_val_path, df_train_c_path]
            fold += 1

        return folds
    else:
        return {0: [train_file, None, contin_path]}


def format_moses_opts(input_file, output_file, assoc_mat, pen, complexity_ratio,
                      div_pressure, scm_path, log_path, alpha, gamma, fs, cpath, target_feature, xopts=None):

    opts = f"{moses_options} -i {input_file} -o {output_file} -f {log_path} " \
           f"--complexity-ratio {complexity_ratio} -u {target_feature} "

    if pen:
        opts = f"{opts} --rel-types IntensionalSimilarity --feature-type ConceptNode --scm-path {scm_path} " \
               f"--inconsistency-gamma {gamma} --inconsistency-alpha {alpha} --assoc-mat {assoc_mat} --contin-path={cpath} "

    if div_pressure > 0:
        opts = f"{opts} --diversity-pressure {div_pressure} "

    if fs:
        opts = f"{opts} --enable-fs=1 --fs-focus=all --fs-algo=smd --fs-target-size=6"

    if xopts is not None:
        opts = f"{opts} {xopts} "

    return opts


def run_moses(mose_opts):
    cmd = ["asmoses"]

    for opt in mose_opts.split():
        cmd.append(opt)

    process = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    return process.returncode, stdout, stderr


def run_exp(train_file, path, seeds, assoc_mat, pen, complexity_ratio, div_pressure,
            scm_path, logger, alpha, gamma, num_folds, fs, contin_path, target, xopts=None):

    folds = split_dataset(train_file, num_folds, path, contin_path, target)

    for seed in seeds:
        seed_path = os.path.join(path, "seed_{0}".format(seed))
        os.makedirs(seed_path)

        for fold, files in folds.items():
            # file format combo_s_XX_f_XX.txt
            outfile = f"combo_s_{seed}_f_{fold}.txt"
            outfile_path = os.path.join(seed_path, outfile)

            logfile = f"log_s_{seed}_f_{fold}.log"
            logfile_path = os.path.join(seed_path, logfile)

            train_file_path, val_file_path, contin_file_path = files[0], files[1], files[2]

            opts = format_moses_opts(train_file_path, outfile_path, assoc_mat, pen, complexity_ratio,
                                     div_pressure, scm_path, logfile_path, alpha, gamma, fs, contin_file_path, target, xopts=xopts)

            opts = f"{opts} --random-seed {seed}"

            logger.debug(f"Running MOSES with the following options\n {opts}")

            code, sout, serr = run_moses(opts)

            if code != 0:
                logger.error(f"MOSES Process returned {code}\n.Std err - {serr}")
                exit(-1)


if __name__ == "__main__":

    args = parse_args()

    # make sure the provided files actually exists
    if not os.path.exists(args.train):
        print(f"Error: train file {args.train} doesn't exist or cannot be found!")
        exit(-1)

    if not os.path.exists(args.dir):
        print(f"Error: working directory {args.dir} doesn't exist or cannot be found! Make sure it exists and writable")
        exit(-1)

    if not os.path.exists(args.seed):
        print(f"Error: seed file {args.seed} doesn't exist or cannot be found!")
        exit(-1)

    if args.inconsistency_pen:
        if args.scm_path is None or args.scm_path == "":
            print("Error: Inconsistency penalty option ON but atomese path not provided. "
                  "Use the -s or --scm_path option to provide path to the atomese file.")
            exit(-1)
        if not os.path.exists(args.scm_path):
            print(f"Error: Atomese file path {args.scm_path} doesn't exist or cannot be found!")
            exit(-1)

        if not os.path.exists(args.mat):
            print(f"Error: Association matrix path {args.mat} doesn't exist or cannot be found!")
            os.exit(-1)

    seeds = []

    with open(args.seed, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    logger = set_up_logging(args.dir)
    logger.info("Starting experiment..")
    start = time.time()
    run_exp(args.train, args.dir, seeds, args.mat, args.inconsistency_pen, args.complexity_ratio,
            args.diversity_pressure, args.scm_path, logger, args.alpha, args.gamma, args.fold, args.fs, args.cpath, args.target, xopts=args.xopts)

    end = time.time()
    elapsed = end - start
    logger.info(f"Experiment Done in {str(datetime.timedelta(seconds=elapsed))}")
