import numpy as np
from data_generator.data_generator import generate_data, price_round
from optimizers.optimization import pricing_optimization
from optimizers.optimizers import (
    ScipyNlpOptimizationModel,
    PyomoNlpOptimizationModel,
    PyomoLpOptimizationModel,
    CvxpyLpOptimizationModel,
)


def calc_metrics(df, tp='cur'):
    sfx = ''
    if tp == 'cur':
        sfx = ''
    elif tp == 'opt':
        sfx = '_opt'
    R_ = sum(df['P' + sfx] * df['Q' + sfx])
    M_ = sum((df['P' + sfx] - df['C']) * df['Q' + sfx])
    return R_, M_


def perc_delta(v_old, v_new, ndigits=2):
    p = round(100. * (v_new / v_old - 1.), ndigits)
    sign = '+' if p >= 0 else '-'
    return sign + str(abs(p)) + '%'


bounds_params = {
    'main_bounds': {
        'lower': 0.9, 'upper': 1.1
    },
    'market_bounds': {
        'lower': 0.85, 'upper': 1.15
    }
}
grid_size = 21
#
data = generate_data(50, bounds_params, grid_size, 0)
# текущая выручка и маржа
R_cur, M_cur = calc_metrics(data['data_nlp'], 'cur')
# параметры для оптимизации
opt_params = {
    'con_mrg': M_cur,
}


def test_solving():
    """
    Тест расчёта на небольших данных
    """

    res_nlp = pricing_optimization(data, PyomoNlpOptimizationModel, opt_params, 'ipopt')
    res_milp = pricing_optimization(data, CvxpyLpOptimizationModel, opt_params, 'GLPK_MI')

    R_opt_nlp, M_opt_nlp = calc_metrics(res_nlp['data'], 'opt')
    R_opt_milp, M_opt_milp = calc_metrics(res_milp['data'], 'opt')

    print('dR = {dR}, dM = {dM} - Изменение выручки и маржи в NLP задаче'.format(
        dR=perc_delta(R_cur, R_opt_nlp), dM=perc_delta(M_cur, M_opt_nlp)))
    print('dR = {dR}, dM = {dM} - Изменение выручки и маржи в MILP задаче'.format(
        dR=perc_delta(R_cur, R_opt_milp),
        dM=perc_delta(M_cur, M_opt_milp)))

    res_nlp['data']['P_opt'] = price_round(res_nlp['data']['P_opt'])
    res_nlp['data']['x_opt'] = res_nlp['data']['P_opt'] / res_nlp['data']['P']
    res_nlp['data']['Q_opt'] = res_nlp['data']['Q'] * np.exp(res_nlp['data']['E'] * (res_nlp['data']['x_opt'] - 1))
    R_opt_nlp_ar, M_opt_nlp_ar = calc_metrics(res_nlp['data'], 'opt')

    print('dR = {dR}, dM = {dM} - Изменение выручки и маржи в NLP задаче после округления'.format(
        dR=perc_delta(R_cur, R_opt_nlp_ar), dM=perc_delta(M_cur, M_opt_nlp_ar)))

    ind = res_nlp['data'].groupby(['plu_line_idx'])['x_opt'].nunique().max() == 1
    print(f'Все товары в линейке имеют одинаковые цены: {ind}')
