import os.path
import pickle
from hpo_util import *
from data_utils import NumpyLoader, NumpyData, preprocess_data
from nn_util import *
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
import optuna
import pathlib

def get_result_df(seeds, save_dir, version):
    res_dfs = []
    for seed in seeds:
        df = pd.read_csv(f"{save_dir}/results/bnn_rf_bg_s_{seed}_v{version}.csv")
        res_dfs.append(df)

    bnn_rf_df = pd.concat(res_dfs, ignore_index=True, axis=0)
    return bnn_rf_df

def cross_val_runs(seeds, X, y, J, version, save_dir, saved_config=False, timeout=60,
                   **configs):
    epochs = configs["epochs"]
    act_fn = configs["act_fn"]
    beta = configs["beta"]
    hidden_sizes = configs["hidden_sizes"]
    M = configs["num_models"]
    lr_0 = configs["lr_0"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    J_zero = np.zeros_like(J)
    for seed in tqdm(seeds):
        bnn_rf_bg_dict = {"seed":[], "model": [], "test_rmse": [], "test_r2_score": []}
        transformer = QuantileTransformer(random_state=seed, output_distribution="normal")
        X_train_outer, X_train, X_val, X_test, \
        y_train_outer, y_train, y_val, y_test, (train_indices, val_indices) = preprocess_data(seed, X, y,
                                                                                              transformer, val_size=0.2, test_size=0.2)
        ### BNN + BG
        if saved_config:
            bnn_bg_config = pickle.load(open(f"{save_dir}/configs/bg_bnn_config_s_{seed}_optuna_v{version}.pkl", "rb"))
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(lambda trial: objective_bg_bnn(trial, seed, X_train, X_val, y_train, y_val, J, beta, hidden_sizes, act_fn, bg=True), timeout=timeout)
            bnn_bg_config = study.best_params
            with open(f"{save_dir}/configs/bg_bnn_config_s_{seed}_optuna_v{version}.pkl", "wb") as fp:
                pickle.dump(bnn_bg_config , fp)
                fp.flush()

        num_cycles = bnn_bg_config["num_cycles"]
        batch_size = bnn_bg_config["batch_size"]
        disc_lr_0 = bnn_bg_config["disc_lr_0"]
        temp, sigma = bnn_bg_config["temp"], bnn_bg_config["sigma"]
        eta, mu = bnn_bg_config["eta"], bnn_bg_config["mu"]
        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size,
                                        shuffle=True, drop_last=True)

        bg_bnn_model, bg_bnn_states, bg_bnn_disc_states = train_bg_bnn_model(seed, outer_data_loader, epochs, num_cycles, beta, M, lr_0, disc_lr_0,
                                                                             hidden_sizes, temp, sigma, eta, mu, J, act_fn, show_pgbar=False)

        bnn_bg_rmse_test, bnn_bg_r2_test = score_bg_bnn_model(bg_bnn_model, X_test, y_test, bg_bnn_states,
                                                              bg_bnn_disc_states, True)

        ### BNN w/o BG
        if saved_config:
            bnn_config = pickle.load(open(f"{save_dir}/configs/bnn_config_s_{seed}_optuna_v{version}.pkl", "rb"))
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(lambda trial: objective_bg_bnn(trial, seed, X_train, X_val, y_train, y_val, J_zero, beta, hidden_sizes, act_fn, bg=False), timeout=timeout)
            bnn_config = study.best_params
            with open(f"{save_dir}/configs/bnn_config_s_{seed}_optuna_v{version}.pkl", "wb") as fp:
                pickle.dump(bnn_config, fp)
                fp.flush()

        num_cycles = bnn_config["num_cycles"]
        batch_size = bnn_config["batch_size"]
        disc_lr_0 = bnn_config["disc_lr_0"]
        temp, sigma = bnn_config["temp"], bnn_config["sigma"]
        eta, mu = 1.0, bnn_config["mu"]
        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size,
                                        shuffle=True, drop_last=True)

        bnn_model, bnn_states, bnn_disc_states = train_bg_bnn_model(seed, outer_data_loader, epochs, num_cycles, beta, M, lr_0, disc_lr_0,
                                                                    hidden_sizes, temp, sigma, eta, mu, J_zero, act_fn, show_pgbar=False)

        bnn_rmse_test, bnn_r2_test = score_bg_bnn_model(bnn_model, X_test, y_test, bnn_states, bnn_disc_states, True)


        if os.path.exists(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl"):
            rf_model = pickle.load(open(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl", "rb"))
        else:
            rf_model = train_rf_model(seed, X_train_outer, y_train_outer, train_indices, val_indices)
            pickle.dump(rf_model, open(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl", "wb"))

        rmse_test_rf, r2_test_rf = eval_rf_model(rf_model, X_test, y_test)

        bnn_rf_bg_dict["seed"].append(seed)
        bnn_rf_bg_dict["model"].append("RF")
        bnn_rf_bg_dict["test_rmse"].append(rmse_test_rf)
        bnn_rf_bg_dict["test_r2_score"].append(r2_test_rf)

        bnn_rf_bg_dict["seed"].append(seed)
        bnn_rf_bg_dict["model"].append("BNN w/o BG")
        bnn_rf_bg_dict["test_rmse"].append(bnn_rmse_test)
        bnn_rf_bg_dict["test_r2_score"].append(bnn_r2_test)

        bnn_rf_bg_dict["seed"].append(seed)
        bnn_rf_bg_dict["model"].append("BNN + BG")
        bnn_rf_bg_dict["test_rmse"].append(bnn_bg_rmse_test)
        bnn_rf_bg_dict["test_r2_score"].append(bnn_bg_r2_test)

        # print(f"RF scores - rmse: {rmse_test_rf}, r2: {r2_test_rf}")
        # print(f"BNN w/o scores - rmse: {bnn_rmse_test}, r2: {bnn_r2_test}")
        # print(f"BNN + BG scores - rmse: {bnn_bg_rmse_test} {bnn_bg_r2_test}")

        with open(f"{save_dir}/results/bnn_rf_bg_s_{seed}_v{version}.csv", "w") as fp:
            pd.DataFrame(bnn_rf_bg_dict).to_csv(fp, index=False)
            fp.flush()



    return print("Done")


def run_multiple_drugs(seeds, gene_expr_df, response_df,
                       subset_gene_lst, drug_ids, J, version, gdsc_dir, hp_configs, timeout):

    for drug_id in drug_ids:
        drug_response_data = response_df[response_df["DRUG_ID"] == drug_id]
        drug_name = drug_response_data["DRUG_NAME"].iloc[0].lower()
        drug_exp_response = pd.merge(gene_expr_df, drug_response_data["LN_IC50"], left_index=True, right_index=True)
        print(f"Starting exp for Drug id: {drug_id}/{drug_name}")
        print(f"Total samples for drug {drug_id}/{drug_name}: {drug_exp_response.shape[0]}")

        save_dir = f"{gdsc_dir}/{drug_name}"
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{save_dir}/results").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{save_dir}/configs").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{save_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{save_dir}/dropout").mkdir(parents=True, exist_ok=True)

        X, target = drug_exp_response.iloc[:,:-1], drug_exp_response.iloc[:,-1]
        target = -np.log10(np.exp(target))
        X_selected = X[subset_gene_lst]
        cross_val_runs(seeds, X_selected, target, J, version, save_dir, saved_config=False,
                       timeout=timeout,
                       **hp_configs)
        print(f"Done for drug: {drug_id}/{drug_name}")


def zero_out_ranking(seeds, X, y, J, num_feats, version, save_dir, dropout=False, **configs):
    epochs = configs["epochs"]
    num_cycles = configs["num_cycles"]
    batch_size = configs["batch_size"]
    act_fn = configs["act_fn"]
    beta = configs["beta"]
    lr_0, disc_lr_0 = configs["lr_0"], configs["disc_lr_0"]
    hidden_sizes = configs["hidden_sizes"]
    temp, sigma = configs["temp"], configs["sigma"]
    M = configs["num_models"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    J_zero = np.zeros_like(J)
    p = X.shape[1]
    for seed in tqdm(seeds):
        bg_bnn_rf_res_dict = {"seed": [], "model": [], "num_feats": [], "test_rmse_score": []}

        transformer = QuantileTransformer(random_state=seed, output_distribution="normal")
        X_train_outer, X_train, X_val, X_test, \
        y_train_outer, y_train, y_val, y_test, (train_indices, val_indices) = preprocess_data(seed, X, y,
                                                                                              transformer, val_size=0.2, test_size=0.2)
        ### BNN + BG

        bnn_bg_config = pickle.load(open(f"{save_dir}/configs/bg_bnn_config_s_{seed}_optuna_v{version}.pkl", "rb"))

        eta, mu = bnn_bg_config["eta_sign"]*bnn_bg_config["eta"], bnn_bg_config["mu"]
        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True)

        bg_bnn_model, bg_bnn_states, bg_bnn_disc_states = train_bg_bnn_model(seed, outer_data_loader, epochs, num_cycles, beta, M, lr_0, disc_lr_0,
                                                                             hidden_sizes, temp, sigma, eta, mu, J, act_fn, show_pgbar=False)
        params_bg_bnn = tree_utils.tree_stack(bg_bnn_states)
        gammas_bg_bnn = tree_utils.tree_stack((bg_bnn_disc_states))

        if dropout:
            if os.path.exists(f"{save_dir}/dropout/bg_bnn_dropout_loss_s_{seed}_v{version}.csv"):
                bg_bnn_dropout_loss_df = pd.read_csv(f"{save_dir}/dropout/bg_bnn_dropout_loss_s_{seed}_v{version}.csv")
            else:
                bg_bnn_dropout_loss_df = get_feats_dropout_loss(bg_bnn_model, params_bg_bnn, gammas_bg_bnn, X_train_outer, y_train)
                bg_bnn_dropout_loss_df.to_csv(f"{save_dir}/dropout/bg_bnn_dropout_loss_s_{seed}_v{version}.csv", index=False)

            bg_bnn_feat_idx = bg_bnn_dropout_loss_df["feats_idx"].to_list()

        else:
            igs_bg = jax.vmap(integrated_gradients, in_axes=(None, None, None, 0, None))(bg_bnn_model, params_bg_bnn, gammas_bg_bnn, X_train_outer, 20).squeeze()
            bg_bnn_feat_idx = np.argsort(np.mean(np.abs(igs_bg), axis=0))[::-1]

        ##### BNN w/o BG
        bnn_config = pickle.load(open(f"{save_dir}/configs/bnn_config_s_{seed}_optuna_v10b.pkl", "rb"))

        eta, mu = 1.0, bnn_config["mu"]
        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True)

        bnn_model, bnn_states, bnn_disc_states = train_bg_bnn_model(seed, outer_data_loader, epochs, num_cycles, beta, M, lr_0, disc_lr_0,
                                                                    hidden_sizes, temp, sigma, eta, mu, J_zero, act_fn, show_pgbar=False)

        params_bnn = tree_utils.tree_stack(bnn_states)
        gammas_bnn = tree_utils.tree_stack(bnn_disc_states)

        if dropout:
            if os.path.exists(f"{save_dir}/dropout/bnn_dropout_loss_s_{seed}_v{version}.csv"):
                bnn_dropout_loss_df = pd.read_csv(f"{save_dir}/dropout/bnn_dropout_loss_s_{seed}_v{version}.csv")
            else:
                bnn_dropout_loss_df = get_feats_dropout_loss(bnn_model, params_bnn, gammas_bnn, X_train_outer, y_train_outer)
                bnn_dropout_loss_df.to_csv(f"{save_dir}/dropout/bnn_dropout_loss_s_{seed}_v{version}.csv", index=False)

            bnn_feat_idx = bnn_dropout_loss_df["feats_idx"].to_list()

        else:
            igs_bnn = jax.vmap(integrated_gradients, in_axes=(None, None, None, 0, None))(bnn_model, params_bnn, gammas_bnn, X_train_outer, 20).squeeze()
            bnn_feat_idx = np.argsort(np.mean(np.abs(igs_bnn), axis=0))[::-1]

        if os.path.exists(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl"):
            rf_model = pickle.load(open(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl", "rb"))
        else:
            rf_model = train_rf_model(seed, X_train_outer, y_train_outer, train_indices, val_indices)
            pickle.dump(rf_model, open(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl", "wb"))

        rf_feat_idx = np.argsort(rf_model.feature_importances_)[::-1]

        for num_feat in num_feats:

            ### BNN + BG
            rmse_bg_bnn = zero_out_score(bg_bnn_model, X_test, y_test, bg_bnn_states, bg_bnn_disc_states, num_feat, bg_bnn_feat_idx, False)

            ## BNN w/o BG
            rmse_bnn = zero_out_score(bnn_model, X_test, y_test, bnn_states, bnn_disc_states, num_feat, bnn_feat_idx, False)

            ## RF
            rf_mask = np.zeros(p)
            rf_mask[rf_feat_idx[:num_feat]] = 1.0
            X_test_rf_m = X_test @ np.diag(rf_mask)
            rmse_rf, _ = eval_rf_model(rf_model, X_test_rf_m, y_test)

            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("RF")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_rf)

            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("BNN")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_bnn)


            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("BNN + BG")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_bg_bnn)

        if dropout:
            pd.DataFrame(bg_bnn_rf_res_dict).to_csv(f"{save_dir}/results/feat_zero_out_comp_bnn_bg_rf_s_{seed}_v{version}.csv", index=False)
        else:
            pd.DataFrame(bg_bnn_rf_res_dict).to_csv(f"{save_dir}/results/feat_zero_out_comp_bnn_bg_rf_s_{seed}_ig_v{version}.csv", index=False)

    print("Done")


def retrain_with_mlp(seeds, X, y, J, num_feats, version, save_dir, dropout=False,
                     **configs):
    epochs = configs["epochs"]
    num_cycles = configs["num_cycles"]
    batch_size = configs["batch_size"]
    act_fn = configs["act_fn"]
    beta = configs["beta"]
    lr_0, disc_lr_0 = configs["lr_0"], configs["disc_lr_0"]
    hidden_sizes = configs["hidden_sizes"]
    temp, sigma = configs["temp"], configs["sigma"]
    M = configs["num_models"]

    mlp_lr = configs["mlp_config"]["lr_0"]
    mlp_epochs = configs["mlp_config"]["epochs"]
    mlp_hidden_sizes = configs["mlp_config"]["hidden_sizes"]
    mlp_act_fn = configs["mlp_config"]["act_fn"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    J_zero = np.zeros_like(J)
    p = X.shape[1]
    for seed in tqdm(seeds):
        bg_bnn_rf_res_dict = {"seed": [], "model": [], "num_feats": [], "test_rmse_score": []}

        transformer = QuantileTransformer(random_state=seed, output_distribution="normal")
        X_train_outer, X_train, X_val, X_test, \
        y_train_outer, y_train, y_val, y_test, (train_indices, val_indices) = preprocess_data(seed, X, y,
                                                                                              transformer, val_size=0.2, test_size=0.2)
        ### BNN + BG

        bnn_bg_config = pickle.load(open(f"{save_dir}/configs/bg_bnn_config_s_{seed}_optuna_v{version}.pkl", "rb"))

        eta, mu = bnn_bg_config["eta_sign"]*bnn_bg_config["eta"], bnn_bg_config["mu"]
        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True)

        bg_bnn_model, bg_bnn_states, bg_bnn_disc_states = train_bg_bnn_model(seed, outer_data_loader, epochs, num_cycles, beta, M, lr_0, disc_lr_0,
                                                                             hidden_sizes, temp, sigma, eta, mu, J, act_fn, show_pgbar=False)
        params_bg_bnn = tree_utils.tree_stack(bg_bnn_states)
        gammas_bg_bnn = tree_utils.tree_stack((bg_bnn_disc_states))

        if dropout:
            if os.path.exists(f"{save_dir}/dropout/bg_bnn_dropout_loss_s_{seed}_v{version}.csv"):
                bg_bnn_dropout_loss_df = pd.read_csv(f"{save_dir}/dropout/bg_bnn_dropout_loss_s_{seed}_v{version}.csv")
            else:
                bg_bnn_dropout_loss_df = get_feats_dropout_loss(bg_bnn_model, params_bg_bnn, gammas_bg_bnn, X_train_outer, y_train)
                bg_bnn_dropout_loss_df.to_csv(f"{save_dir}/dropout/bg_bnn_dropout_loss_s_{seed}_v{version}.csv", index=False)

            bg_bnn_feat_idx = bg_bnn_dropout_loss_df["feats_idx"].to_list()

        else:
            igs_bg = jax.vmap(integrated_gradients, in_axes=(None, None, None, 0, None))(bg_bnn_model, params_bg_bnn, gammas_bg_bnn, X_train_outer, 20).squeeze()
            bg_bnn_feat_idx = np.argsort(np.mean(np.abs(igs_bg), axis=0))[::-1]

        ##### BNN w/o BG
        bnn_config = pickle.load(open(f"{save_dir}/configs/bnn_config_s_{seed}_optuna_v{version}.pkl", "rb"))

        eta, mu = 1.0, bnn_config["mu"]
        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True)

        bnn_model, bnn_states, bnn_disc_states = train_bg_bnn_model(seed, outer_data_loader, epochs, num_cycles, beta, M, lr_0, disc_lr_0,
                                                                    hidden_sizes, temp, sigma, eta, mu, J_zero, act_fn, show_pgbar=False)

        params_bnn = tree_utils.tree_stack(bnn_states)
        gammas_bnn = tree_utils.tree_stack(bnn_disc_states)

        if dropout:
            if os.path.exists(f"{save_dir}/dropout/bnn_dropout_loss_s_{seed}_v{version}.csv"):
                bnn_dropout_loss_df = pd.read_csv(f"{save_dir}/dropout/bnn_dropout_loss_s_{seed}_v{version}.csv")
            else:
                bnn_dropout_loss_df = get_feats_dropout_loss(bnn_model, params_bnn, gammas_bnn, X_train_outer, y_train_outer)
                bnn_dropout_loss_df.to_csv(f"{save_dir}/dropout/bnn_dropout_loss_s_{seed}_v{version}.csv", index=False)

            bnn_feat_idx = bnn_dropout_loss_df["feats_idx"].to_list()

        else:
            igs_bnn = jax.vmap(integrated_gradients, in_axes=(None, None, None, 0, None))(bnn_model, params_bnn, gammas_bnn, X_train_outer, 20).squeeze()
            bnn_feat_idx = np.argsort(np.mean(np.abs(igs_bnn), axis=0))[::-1]

        if os.path.exists(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl"):
            rf_model = pickle.load(open(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl", "rb"))
        else:
            rf_model = train_rf_model(seed, X_train_outer, y_train_outer, train_indices, val_indices)
            pickle.dump(rf_model, open(f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl", "wb"))

        rf_feat_idx = np.argsort(rf_model.feature_importances_)[::-1]

        for num_feat in num_feats:

            bg_bnn_sel_fts = bg_bnn_feat_idx[:num_feat]
            bnn_sel_fts = bnn_feat_idx[:num_feat]
            rf_sel_fts = rf_feat_idx[:num_feat]

            X_train_sel_bg_bnn = X_train_outer[:,bg_bnn_sel_fts]
            X_train_sel_bnn = X_train_outer[:,bnn_sel_fts]
            X_train_sel_rf = X_train_outer[:,rf_sel_fts]

            X_test_sel_bg_bnn = X_test[:,bg_bnn_sel_fts]
            X_test_sel_bnn = X_test[:,bnn_sel_fts]
            X_test_sel_rf = X_test[:,rf_sel_fts]

            bg_bnn_data_loader = NumpyLoader(NumpyData(X_train_sel_bg_bnn, y_train_outer),
                                                    batch_size=batch_size, shuffle=True)

            bnn_data_loader = NumpyLoader(NumpyData(X_train_sel_bnn, y_train_outer),
                                             batch_size=batch_size, shuffle=True)

            rf_data_loader = NumpyLoader(NumpyData(X_train_sel_rf, y_train_outer),
                                             batch_size=batch_size, shuffle=True)

            ### BNN + BG
            bg_mlp_model, bg_bnn_param_mlp = train_mlp_model(seed, bg_bnn_data_loader, mlp_epochs, mlp_lr,
                                                mlp_hidden_sizes, mlp_act_fn, show_pgbar=False)

            rmse_bg_bnn = eval_mlp_model(bg_mlp_model, X_test_sel_bg_bnn, y_test, bg_bnn_param_mlp)

            ## BNN w/o BG
            bnn_mlp_model, bnn_param_mlp = train_mlp_model(seed, bnn_data_loader, mlp_epochs, mlp_lr,
                                                           mlp_hidden_sizes, mlp_act_fn, show_pgbar=False)

            rmse_bnn = eval_mlp_model(bnn_mlp_model, X_test_sel_bnn, y_test, bnn_param_mlp)

            ## RF
            rf_mlp_model, rf_param_mlp = train_mlp_model(seed, rf_data_loader, mlp_epochs, mlp_lr,
                                                         mlp_hidden_sizes, mlp_act_fn, show_pgbar=False)

            rmse_rf = eval_mlp_model(rf_mlp_model, X_test_sel_rf, y_test, rf_param_mlp)

            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("RF")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_rf)

            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("BNN")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_bnn)


            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("BNN + BG")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_bg_bnn)

        if dropout:
            pd.DataFrame(bg_bnn_rf_res_dict).to_csv(f"{save_dir}/results/feat_retrain_comp_bnn_bg_rf_s_{seed}_v{version}.csv", index=False)
        else:
            pd.DataFrame(bg_bnn_rf_res_dict).to_csv(f"{save_dir}/results/feat_retrain_comp_bnn_bg_rf_s_{seed}_ig_v{version}.csv", index=False)

    print("Done")