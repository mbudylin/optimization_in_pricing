# -*- coding: utf-8 -*-
import os
import sys
from itertools import product
import pandas as pd
import argparse

sys.path.append('./OptimizersArticle')

from data_generator.data_generator import generate_data

from optimizers.optimizers import (
    ScipyNlpOptimizationModel,
    PyomoNlpOptimizationModel,
    PyomoLpOptimizationModel,
    CvxpyLpOptimizationModel
)
from optimizers.optimization import pricing_optimization

import os

IS_DOCKER = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)
if IS_DOCKER:
    STAT_PATH = './data/docker/stat'
else:
    STAT_PATH = './data/stat'


SEED_GRID = list(range(25))
N_GRID = [10, 20, 50, 100, 200, 500, 1000]
GRID = product(N_GRID, SEED_GRID)
BOUNDS_PARAMS = {
    'main_bounds': {
        'lower': 0.9, 'upper': 1.1
    },
    'market_bounds': {
        'lower': 0.85, 'upper': 1.15
    }
}
LP_MAX_GRID_SIZE = 21

def get_files(file_path):
    files = []
    if os.path.exists(file_path):
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith('.csv')]
    return files

def commit(statuses, times, solvers, opt_types, optimization_results, solver_name, opt_type):
    """
    Запись результатов расчётов
    """

    # Если данных расчёта нет, то пропустить
    if len(optimization_results) == 0:
        return

    statuses.append('ok' if optimization_results['status'] == '0' else optimization_results['status'])
    times.append(optimization_results['t'])
    solvers.append(solver_name)
    opt_types.append(opt_type)
    print(f"{solver_name} finish \t{optimization_results['t']}")

def calculate_optimizers(N, seed):

    statuses, times, solvers, opt_types = [], [], [], []

    print('--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--')
    print(N, seed)

    # генерация данных для NLP и LP оптимизации
    data = generate_data(N, BOUNDS_PARAMS, LP_MAX_GRID_SIZE, seed)

    M_cur = sum(data['data_nlp']['Q'] * (data['data_nlp']['P'] - data['data_nlp']['C']))

    opt_params = {
        'add_con_mrg': M_cur,
    }

    if N < 1000:
        res = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'slsqp')
        commit(statuses, times, solvers, opt_types, res, 'scipy.slsqp', 'NLP')

    if N < 500:
        res = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'cobyla')
        commit(statuses, times, solvers, opt_types, res, 'scipy.cobyla', 'NLP')

    if N < 500:
        res = pricing_optimization(data, ScipyNlpOptimizationModel, opt_params, 'trust-constr')
        commit(statuses, times, solvers, opt_types, res, 'scipy.trust-constr', 'NLP')

    res = pricing_optimization(data, PyomoNlpOptimizationModel, opt_params, 'ipopt')
    commit(statuses, times, solvers, opt_types, res, 'pyomo.ipopt', 'NLP')

    res = pricing_optimization(data, PyomoLpOptimizationModel, opt_params, 'cbc')
    commit(statuses, times, solvers, opt_types, res, 'pyomo.cbc', 'MILP')

    res = pricing_optimization(data, PyomoLpOptimizationModel, opt_params, 'glpk')
    commit(statuses, times, solvers, opt_types, res, 'pyomo.glpk', 'MILP')

    res = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'CBC')
    commit(statuses, times, solvers, opt_types, res, 'cvxpy.cbc', 'MILP')

    res = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'GLPK_MI')
    commit(statuses, times, solvers, opt_types, res, 'cvxpy.glpk', 'MILP')

    if N < 2000:
        res = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'ECOS_BB')
        commit(statuses, times, solvers, opt_types, res, 'cvxpy.ecos', 'MILP')

    return statuses, times, solvers, opt_types

def save_stat(full_name, N, seed, statuses, times, solvers, opt_types):

    df = pd.DataFrame({
        'N': N,
        'seed': seed,
        'solver': solvers,
        'opt_type': opt_types,
        't': times,
        'status': statuses
    })
    df.to_csv(full_name, index=False)

def optimizers_calc_stat(grid, file_path_stat, overwrite, N_bash, seed_bash):

    existed_files = set(get_files(file_path_stat))

    if (N_bash is not None) and (seed_bash is not None) and overwrite:
        file_name = 's_' + str(N_bash) + '_' + str(seed_bash) + '.csv'
        statuses, times, solvers, opt_types = calculate_optimizers(N_bash, seed_bash)
        full_name = os.path.join(file_path_stat, file_name)
        print(full_name)
        save_stat(full_name, N_bash, seed_bash, statuses, times, solvers, opt_types)
        return

    for N, seed in grid:

        file_name = 's_' + str(N) + '_' + str(seed) + '.csv'

        if not overwrite:
            if file_name in existed_files:
                continue

        statuses, times, solvers, opt_types = calculate_optimizers(N, seed)

        full_name = os.path.join(file_path_stat, file_name)

        print(full_name)
        save_stat(full_name, N, seed, statuses, times, solvers, opt_types)


def optimizers_collect_stat(file_path_stat):
    files_stat = get_files(file_path_stat)
    df = pd.concat([pd.read_csv(os.path.join(file_path_stat, file_name)) for file_name in files_stat])
    return df


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-ow", "--overwrite", required=False, default=False, action='store_true',
                             help='Перезапись файлов со статистикой отработки оптимизаторов')
    args_parser.add_argument("-nb", "--N_bash", required=False, default=None, type=int,
                             help='Размер задачи')
    args_parser.add_argument("-sb", "--seed_bash", required=False, default=None, type=int,
                             help='Seed')

    args = vars(args_parser.parse_args())

    optimizers_calc_stat(GRID, STAT_PATH, overwrite=args['overwrite'], N_bash=args['N_bash'], seed_bash=args['seed_bash'])
    stat_df = optimizers_collect_stat(STAT_PATH)
