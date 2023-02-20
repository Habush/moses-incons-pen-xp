from gplearn.genetic import SymbolicTransformer, SymbolicClassifier, SymbolicRegressor
from gplearn.functions import make_function
import operator
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
import pandas as pd
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_best_programs(gp, num_models, classifier=True, ascending=True, sort_fit="OOB_fitness"):
    gp_dict = {'Gen': [], "Ind": [], "Fitness": [], 'OOB_fitness': [], "Equation": []}

    if classifier:
        for idGen in range(len(gp._programs)):
            for idPopulation in range(gp.population_size):
                gp_dict["Gen"].append(idGen)
                gp_dict["Ind"].append(idPopulation)
                gp_dict["Fitness"].append(gp._programs[idGen][idPopulation].fitness_)
                gp_dict["OOB_fitness"].append(gp._programs[idGen][idPopulation].oob_fitness_)
                gp_dict["Equation"].append(str(gp._programs[idGen][idPopulation]))
    else:
        for idx, prog in enumerate(gp._programs[-1]):
            gp_dict["Gen"].append(-1)
            gp_dict["Ind"].append(idx)
            gp_dict["Fitness"].append(prog.fitness_)
            gp_dict["OOB_fitness"].append(prog.oob_fitness_)
            gp_dict["Equation"].append(str(prog))

    gp_df = pd.DataFrame(gp_dict).sort_values(sort_fit, ascending=ascending)[:num_models]
    programs = []
    for i in range(num_models):
        gen, ind = int(gp_df.iloc[i]["Gen"]), int(gp_df.iloc[i]["Ind"])
        programs.append(gp._programs[gen][ind])

    return programs, gp_df


def gp_transform(est, X, classifier=False, num_models=100, sort_fit="Fitness"):
    if classifier or (sort_fit == "OOB_fitness"):
        programs, gp_df = get_best_programs(est, num_models, classifier, sort_fit=sort_fit, ascending=classifier)
        out = np.zeros((X.shape[0], len(programs)))
        for i, prog in enumerate(programs):
            out[:, i] = prog.execute(X)

        return out, gp_df
    else:
        return est.transform(X), None

function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos']

def train_linear_model(seed, X_train, X_test, y_train, y_test):
    cv = KFold(n_splits=3, random_state=seed, shuffle=True)
    param_grid = {"alpha": np.logspace(-3, 2, 20)}
    grid_cv = GridSearchCV(estimator=Ridge(max_iter=10000), param_grid=param_grid,
                           verbose=0, scoring="r2", cv=cv).fit(X_train, y_train)
    lin_model = Ridge(max_iter=10000, **grid_cv.best_params_)
    lin_model.fit(X_train, y_train)
    y_test_pred = lin_model.predict(X_test)

    test_rmse_score = np.sqrt(np.mean((y_test - y_test_pred)**2))
    test_r2_score = r2_score(y_test, y_test_pred)
    test_pearson, test_pval = stats.pearsonr(y_test, y_test_pred)

    return lin_model, test_rmse_score, test_r2_score, test_pearson, test_pval

def train_gp(seed, X_train, X_test, y_train, y_test, num_models=5, sort_fit="OOB_fitness", verbose=0, num_gen=100,
             p_cxvr=0.8, p_subt_mut=0.1, p_hmut=0.05, p_pmut=0.1, subsample=0.8, complexity_coef=0.05):
    gp_est = SymbolicTransformer(population_size=1000, hall_of_fame=200, n_components=50, generations=num_gen,
                                 function_set=function_set,
                                 p_crossover=p_cxvr, p_subtree_mutation=p_subt_mut,
                                 p_hoist_mutation=p_hmut, p_point_mutation=p_pmut,
                                 max_samples=subsample, verbose=verbose,
                                 parsimony_coefficient=complexity_coef, random_state=seed)

    gp_est.fit(X_train, y_train)

    gp_features_train, gp_train_df = gp_transform(gp_est, X_train, classifier=False, sort_fit=sort_fit, num_models=num_models)
    gp_features_test, gp_test_df = gp_transform(gp_est, X_test, classifier=False, sort_fit=sort_fit, num_models=num_models)

    X_train_comb = np.concatenate([X_train, gp_features_train], axis=1)
    X_test_comb = np.concatenate([X_test, gp_features_test], axis=1)


    test_rmse_score, test_r2_score, test_pearson, test_pval = train_linear_model(seed, X_train_comb, X_test_comb, y_train, y_test)

    return test_rmse_score, test_r2_score, test_pearson, test_pval, gp_test_df

def train_gp_v2(seed, X_train, X_test, y_train,y_test, verbose=0, num_gen=100,
                p_cxvr=0.8, p_subt_mut=0.1, p_hmut=0.05, p_pmut=0.1, subsample=0.8, complexity_coef=0.05):
    gp_est = SymbolicRegressor(population_size=1000, generations=num_gen,
                               function_set=function_set,
                               p_crossover=p_cxvr, p_subtree_mutation=p_subt_mut,
                               p_hoist_mutation=p_hmut, p_point_mutation=p_pmut,
                               max_samples=subsample, verbose=verbose,
                               parsimony_coefficient=complexity_coef, random_state=seed)

    gp_est.fit(X_train, y_train)

    y_test_pred = gp_est.predict(X_test)
    test_rmse_score = np.sqrt(np.mean((y_test - y_test_pred)**2))
    test_r2_score = r2_score(y_test, y_test_pred)
    test_pearson, test_pval = stats.pearsonr(y_test, y_test_pred)

    return test_rmse_score, test_r2_score, test_pearson, test_pval, gp_est