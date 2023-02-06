from typing import Dict, Any
from time import time
from functools import wraps
import pandas as pd


def timeit(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        duration = time() - t0
        result['t'] = duration
        return result
    return wrapper


@timeit
def pricing_optimization(sources: Dict,
                         opt_model: Any,
                         opt_params: Dict,
                         solver: str,
                         solver_option: Dict = {},
                         ) -> Dict:
    """
    Запуск расчета оптимальных цен с помощью указанного класса оптимизатора и параметров
    :param sources: входные данные для оптимизации
    :param opt_model: класс модели оптимизатора
    :param opt_params: параметры оптимизации
    :param solver: солвер для оптимизации
    :param solver_option: параметры солвера
    :return: словарь, возвращаемый моделью оптимизации
    """

    model = opt_model(sources)

    model.init_variables()
    model.init_objective()
    model.add_constraints(opt_params)
    result = model.solve(solver=solver, options=solver_option)
    result['model_class'] = model
    return result


@timeit
def pricing_optimization_simple(data: pd.DataFrame,
                                opt_model: Any,
                                solver: str,
                                ) -> Dict:
    """
    Запуск расчета оптимальных цен с помощью указанного класса оптимизатора и параметров
    :param sources: входные данные для оптимизации
    :param opt_model: класс модели оптимизатора
    :param opt_params: параметры оптимизации
    :param solver: солвер для оптимизации
    :param solver_option: параметры солвера
    :return: словарь, возвращаемый моделью оптимизации
    """

    model = opt_model(data)

    model.init_objective()
    model.add_constraints()
    model.init_constraints()
    result = model.solve(solver=solver)
    result['model_class'] = model
    return result
