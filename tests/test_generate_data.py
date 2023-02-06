from data_generator.data_generator import generate_data

N = 10
GRID_MAX_SIZE = 21
BOUNDS_PARAMS = {
    'main_bounds': {
        'lower': 0.9, 'upper': 1.1
    },
    'market_bounds': {
        'lower': 0.85, 'upper': 1.15
    }
}


def test_generate_data():
    """
    проверка сгенерированных данных
    """
    data = generate_data(N, BOUNDS_PARAMS, GRID_MAX_SIZE, 0)

    # проверка что есть все поля в генерируемых данных
    keys = set(data.keys())
    need_keys = {'data_nlp', 'data_milp', 'plu_idx_in_line'}
    keys_diff = need_keys - keys
    assert len(keys_diff) == 0, 'не хватает поле в data: %s' % keys_diff
    # проверка наличия необходимых полей в data_nlp
    data_nlp = data['data_nlp']
    cols_nlp = set(data_nlp.columns)
    need_cols_nlp = {'plu_idx', 'plu_line_idx', 'P', 'PC', 'Q', 'C', 'x_lower', 'x_upper'}
    cols_nlp_diff = need_cols_nlp - cols_nlp
    assert len(cols_nlp_diff) == 0, 'в data_nlp не хватает колонок: %s' % cols_nlp_diff
    # проверка наличия необходимых полей в data_milp
    data_milp = data['data_milp']
    cols_milp = set(data_milp.columns)
    need_cols_milp = {'plu_line_idx', 'Ps', 'Qs', 'C', 'xs', 'grid_size', 'P_idx', 'n_plu'}
    cols_milp_diff = need_cols_milp - cols_milp
    assert len(cols_milp_diff) == 0, 'в data_nlp не хватает колонок: %s' % cols_milp_diff
