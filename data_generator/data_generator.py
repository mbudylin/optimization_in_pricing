from typing import Dict
import numpy as np
import pandas as pd


def price_round(prices):
    """
    Округление до ближайшего числа вида XXX.99
    """
    prices_rounded = np.round(prices + 0.01) - 0.01
    return prices_rounded


def generate_simple_data(N, seed=0):
    """
    Генерация данных для тестирования оптимизационной модели
    """
    np.random.seed(seed)

    data = pd.DataFrame({'plu': range(1, N + 1)})
    data['P'] = price_round((np.random.gamma(2., 3., N) + 4.) * 10.)
    P_mean = data['P'].mean()
    data['Q'] = np.random.chisquare(5., N) * np.exp(-data['P'] / P_mean)
    data['E'] = -np.random.gamma(1.7, 0.9, N)

    data['PC'] = price_round(
        data['P'] * np.random.normal(1.0, 0.2 * np.exp(-data['P'] / P_mean))
    )
    data['C'] = round(data['P'] / np.random.normal(1.28, 0.2, N), 2)

    data['x_lower'] = 0.85 * data['PC'] / data['P']
    data['x_upper'] = 1.15 * data['PC'] / data['P']

    data.drop('PC', axis=1, inplace=True)

    return data


def generate_base_data(N_plu_line, seed=0):
    """
    Генерация данных для тестирования оптимизационной модели
    """
    np.random.seed(seed)

    plu_line = list(range(N_plu_line))
    plu_cnt_in_line = np.random.poisson(0.29, N_plu_line) + 1
    N_plu = sum(plu_cnt_in_line)

    data = pd.DataFrame({'plu_line_idx': plu_line})
    data['plu_cnt'] = plu_cnt_in_line
    data['P'] = price_round((np.random.gamma(2., 3., N_plu_line) + 4.) * 10.)
    P_mean = data['P'].mean()
    data['Q'] = np.random.chisquare(5., N_plu_line) * np.exp(-data['P'] / P_mean)
    data['E'] = -np.random.gamma(1.7, 0.9, N_plu_line)
    data['PC'] = price_round(
        data['P'] * np.random.normal(1.0, 0.2 * np.exp(-data['P'] / P_mean))
    )
    data['C'] = round(data['P'] / np.random.normal(1.28, 0.2, N_plu_line), 2)
    data = data.loc[data.index.repeat(data['plu_cnt'])].reset_index(drop=True)
    data['E_rc'] = abs(np.random.normal(1., 0.4, N_plu))
    data['E'] = np.where(data['plu_cnt'] > 1.0, data['E'] * data['E_rc'], data['E'])
    data.drop(columns=['plu_cnt', 'E_rc'], inplace=True)
    data['plu_idx'] = list(range(N_plu))

    return data


def construct_bounds(df, bounds_params):
    """
    (1) Базовое ограничение изменения цен в +- main_bounds от текущей
    (2) Цена должна попадать в диапазон +- market_bounds от цены конкурента/рыночной цены
    Если диапазоны не накладываются то приоритет у (1)
    """

    df['x_bnd_lower'] = bounds_params['main_bounds']['lower']
    df['x_bnd_upper'] = bounds_params['main_bounds']['upper']

    df['x_lower'] = bounds_params['market_bounds']['lower'] * df['PC'] / df['P']
    df['x_upper'] = bounds_params['market_bounds']['upper'] * df['PC'] / df['P']

    df['x_lower'] = np.clip(df['x_lower'], df['x_bnd_lower'], df['x_bnd_upper'])
    df['x_upper'] = np.clip(df['x_upper'], df['x_bnd_lower'], df['x_bnd_upper'])

    ix = df['x_upper'] - df['x_lower'] < 1.0e-2
    df.loc[ix, 'x_lower'] = df.loc[ix, 'x_bnd_lower']
    df.loc[ix, 'x_upper'] = df.loc[ix, 'x_bnd_upper']

    # df['x_cur'] = 1.0
    df.drop(columns=['x_bnd_lower', 'x_bnd_upper'], inplace=True)

    return df


def construct_lp_grid(df, bounds_params, grid_max_size=21):
    """
    Формирует сетку значений для задачи ЛП
    """
    df = df.copy().reset_index(drop=True)

    def form_grid_by_row(df_row, bnd_lower_value, bnd_upper_value):
        price_grid = np.clip(np.hstack([
            np.linspace(bnd_lower_value, bnd_upper_value, grid_max_size), df_row['P']
        ]), df_row['x_lower'], df_row['x_upper'])
        # набор сетки цен
        ps = np.unique(
            price_round(np.hstack([df_row['x_lower'], price_grid, df_row['x_upper']]) * df_row['P'])
        )
        # оценки спроса по ценам из сетки
        qs = df_row['Q'] * np.exp(df_row['E'] * (ps / df_row['P'] - 1.))
        # сетка ценовых индексов
        xs = ps / df_row['P']
        # обозначим индекс текущей цены в сетке цен, если текущая не попадает в допустимый диапазон ставим -1
        P_idx = np.where(ps == df_row['P'])[0]
        P_idx = P_idx[0] if len(P_idx) > 0 else -1
        grid_size = len(ps)

        return {'xs': xs, 'Ps': ps, 'Qs': qs, 'P_idx': P_idx, 'grid_size': grid_size}

    bnd_lower = bounds_params['main_bounds']['lower']
    bnd_upper = bounds_params['main_bounds']['upper']

    #     df[['xs', 'Ps', 'Qs', 'P_idx', 'grid_size']] =\
    #         df.apply(lambda x: form_grid_by_row(x, bnd_lower, bnd_upper), axis=1)

    xs_list = [None] * len(df)
    Ps_list = [None] * len(df)
    Qs_list = [None] * len(df)
    P_idx_list = [None] * len(df)
    grid_size_list = [None] * len(df)

    for i, row in df.iterrows():
        res = form_grid_by_row(row, bnd_lower, bnd_upper)
        xs_list[i] = res['xs']
        Ps_list[i] = res['Ps']
        Qs_list[i] = res['Qs']
        P_idx_list[i] = res['P_idx']
        grid_size_list[i] = res['grid_size']
    df['xs'] = xs_list
    df['Ps'] = Ps_list
    df['Qs'] = Qs_list
    df['P_idx'] = P_idx_list
    df['grid_size'] = grid_size_list

    g_max = df['grid_size'].max()

    def pad_col(x):
        return np.pad(x, (0, g_max - len(x)))

    for col in ['xs', 'Ps', 'Qs']:
        df[col] = df[col].apply(pad_col)

    # собираем все необходимые данные по линейкам
    df = df.groupby(['plu_line_idx']).agg(
        P=('P', 'mean'), PC=('PC', 'mean'), C=('C', 'mean'),
        Ps=('Ps', 'mean'), Qs=('Qs', 'sum'), xs=('xs', 'mean'),
        grid_size=('grid_size', 'min'),
        P_idx=('P_idx', 'min'), n_plu=('plu_idx', 'count')
    ).reset_index()

    return df


def generate_data(N_plu_line, bounds_params, grid_max_size, seed=0) -> Dict:
    """
    Генерация модельных данных для NLP и LP задачи
    :param N_plu_line: количество генерируемых линеек товаров
    :param bounds_params: параметры для границ поиска цены
    :param grid_max_size: максимальный размер сетки для поиска цены в задаче LP
    :param seed:
    :return: словарь с полями, содержащие данные для NLP('data_nlp'), LP('data_lp') и
    составом линеек с более чем 1 товаром('plu_idx_in_line') в виде словаря
    """
    data_base = generate_base_data(N_plu_line, seed)
    data_nlp = construct_bounds(data_base, bounds_params)
    data_lp = construct_lp_grid(data_base, bounds_params, grid_max_size)
    plu_idx_in_line = (
        data_base
        .groupby(['plu_line_idx'])
        .agg(n_plu=('plu_idx', 'count'), plu_idx=('plu_idx', lambda x: list(x)))
        .drop(columns=['n_plu'])
        .to_dict()
        ['plu_idx']
    )

    return {
        'data_nlp': data_nlp,
        'data_milp': data_lp,
        'plu_idx_in_line': plu_idx_in_line,
    }
