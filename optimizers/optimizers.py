"""
Классы оптимизаторов.
"""
from typing import Dict
import abc
import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize
import cvxpy as cp
import pyomo.environ as pyo


class OptimizationModel(abc.ABC):
    """
    Базовый класс для оптимизаторов с ЦО
    """

    def __init__(self, data_sources, table_link):

        self.data_sources = data_sources
        self.data = data_sources[table_link].copy()
        self.plu_idx_in_line = data_sources['plu_idx_in_line'].copy()

        if 'plu_idx' in self.data.columns:
            self.plu_idx = self.data['plu_idx'].values

        self.N = self.data['plu_line_idx'].nunique()
        self.N_SIZE = len(self.data['plu_line_idx'])
        self.plu_line_idx = self.data['plu_line_idx'].values
        self.P = self.data['P'].values

        if 'Q' in self.data.columns:
            self.Q = self.data['Q'].values

        if 'E' in self.data.columns:
            self.E = self.data['E'].values

        self.PC = self.data['PC'].values
        self.C = self.data['C'].values

        # границы для индексов
        if 'x_lower' in self.data.columns and 'x_upper' in self.data.columns:
            self.x_lower = self.data['x_lower'].values
            self.x_upper = self.data['x_upper'].values
            self.x_init = 0.5 * (self.x_lower + self.x_upper)

    @abc.abstractmethod
    def init_variables(self):
        """
        Инициализация переменных в модели
        """
        pass

    @abc.abstractmethod
    def init_objective(self):
        """
        Инициализация целевой функции - выручка
        """
        pass

    @abc.abstractmethod
    def add_con_mrg(self, m_min):
        """
        Добавление в модель ограничения на маржу
        """
        pass

    def add_constraints(self, opt_params: Dict):
        """
        Добавление ограничений, если они заданы.
        Название метода должно начинаться с 'add_[название ограничения]'
        Например: 'add_con_mrg' для ограничения con_mrg в параметрах
        """
        for con_name, param in opt_params.items():
            add_method_name = 'add_' + con_name
            add_method = getattr(self, add_method_name, None)
            if add_method is None:
                # такой метод не реализован
                continue
            add_method(param)

    @abc.abstractmethod
    def solve(self, solver, options) -> Dict:
        """
        Метод, запускающий решение поставленной оптимизационной задачи
        """
        pass


class ScipyNlpOptimizationModel(OptimizationModel):
    """
    Класс, который создаёт NLP оптимизационную модель на базе библиотеки scipy
    """

    def __init__(self, data):
        super().__init__(data, 'data_nlp')

        # Задаём объект для модели scipy
        self.obj = None
        self.bounds = None
        self.x0 = None
        self.constraints = {}
        # нормировка для целевой функции
        self.k = 0.01 * sum(self.P * self.Q)

    def _el(self, E, x):
        return np.exp(E * (x - 1.0))

    def init_variables(self):
        self.bounds = np.array([[None] * self.N] * 2, dtype=float)

        for plu_line_idx_, plu_ in self.plu_idx_in_line.items():
            self.bounds[0][plu_line_idx_] = self.x_lower[plu_[0]]
            self.bounds[1][plu_line_idx_] = self.x_upper[plu_[0]]

        self.x0 = 0.5 * (self.bounds[0] + self.bounds[1])
        # self.x0 = np.random.uniform(self.bounds[0], self.bounds[1])
        A = np.eye(self.N, self.N)
        constr_bounds = LinearConstraint(A, self.bounds[0], self.bounds[1])
        self.constraints['var_bounds'] = constr_bounds

    def init_objective(self):
        def objective(x):
            x_ = x[self.plu_line_idx[self.plu_idx]]
            f = -sum(self.P * x_ * self.Q * self._el(self.E, x_))
            return f / self.k

        self.obj = objective

    def add_con_mrg(self, m_min):
        def con_mrg(x):
            x_ = x[self.plu_line_idx[self.plu_idx]]
            m = sum((self.P * x_ - self.C) * self.Q * self._el(self.E, x_)) / self.k
            return m
        constr = NonlinearConstraint(con_mrg, m_min / self.k, np.inf)
        self.constraints['con_mrg'] = constr

    def solve(self, solver='slsqp', options={}):

        result = minimize(self.obj,
                          self.x0,
                          method=solver,
                          constraints=self.constraints.values(),
                          options=options)

        self.data['x_opt'] = result['x'][self.plu_line_idx[self.plu_idx]]
        self.data['P_opt'] = self.data['x_opt'] * self.data['P']
        self.data['Q_opt'] = self.Q * self._el(self.E, self.data['x_opt'])

        status = str(result['status'])
        status = 'ok' if status == '0' and solver == 'slsqp' else status
        status = 'ok' if status == '1' and solver in ('cobyla', 'trust-constr') else status

        return {
            'message': str(result['message']),
            'status': status,
            'model': result,
            'data': self.data
        }


class PyomoNlpOptimizationModel(OptimizationModel):

    def __init__(self, data):
        super().__init__(data, 'data_nlp')

        self.N = len(self.data['plu_idx'])

        # Задаём объект модели pyomo
        self.model = pyo.ConcreteModel()

    def _el(self, i):
        # вспомогательная функция для пересчета спроса при изменении цены
        return pyo.exp(self.E[i] * (self.model.x[i] - 1.0))

    def init_variables(self):
        def bounds_fun(model, i):
            return self.x_lower[i], self.x_upper[i]

        def init_fun(model, i):
            return self.x_init[i]

        self.model.x = pyo.Var(range(self.N), domain=pyo.Reals, bounds=bounds_fun, initialize=init_fun)

        # добавление условия на равенство цен в линейке
        if len(self.plu_idx_in_line) == 0:
            return

        self.model.con_equal = pyo.Constraint(pyo.Any)

        # название ограничения = plu_line_idx
        for con_idx, idxes in self.plu_idx_in_line.items():
            for i in range(1, len(idxes)):
                con_name = str(con_idx) + '_' + str(i)
                self.model.con_equal[con_name] = (self.model.x[idxes[i]] - self.model.x[idxes[i - 1]]) == 0

    def init_objective(self):
        objective = sum(self.P[i] * self.model.x[i] * self.Q[i] * self._el(i) for i in range(self.N))
        self.model.obj = pyo.Objective(expr=objective, sense=pyo.maximize)

    def add_con_mrg(self, m_min):
        con_mrg_expr = sum((self.P[i] * self.model.x[i] - self.C[i]) * self.Q[i] * self._el(i)
                           for i in range(self.N)) >= m_min
        self.model.con_mrg = pyo.Constraint(rule=con_mrg_expr)

    def add_con_chg_cnt(self, nmax=10000, thr_l=0.98, thr_u=1.02, ):
        self.model.ind_l = pyo.Var(range(self.N), domain=pyo.Binary, initialize=0)
        self.model.ind_r = pyo.Var(range(self.N), domain=pyo.Binary, initialize=0)
        self.model.ind_m = pyo.Var(range(self.N), domain=pyo.Binary, initialize=1)

        self.model.x_interval = pyo.Constraint(pyo.Any)
        self.model.con_ind = pyo.Constraint(pyo.Any)
        K = 0.15
        for i in range(self.N):
            self.model.x_interval['l' + str(i)] = self.model.x[i] - K * (1 - self.model.ind_l[i]) <= thr_l
            self.model.x_interval['r' + str(i)] = self.model.x[i] + K * (1 - self.model.ind_r[i]) >= thr_u
            self.model.x_interval['ml' + str(i)] = self.model.x[i] - K * (1 - self.model.ind_m[i]) <= 1.
            self.model.x_interval['mr' + str(i)] = self.model.x[i] + K * (1 - self.model.ind_m[i]) >= 1.
            self.model.con_ind[i] = (self.model.ind_l[i] + self.model.ind_m[i] + self.model.ind_r[i]) == 1

        self.model.con_max_chg = pyo.Constraint(expr=sum(
            self.model.ind_m[i]
            for i in range(self.N)
        ) >= self.N - nmax)

    def solve(self, solver='ipopt', options={}):
        solver = pyo.SolverFactory(solver, tee=False)
        for option_name, option_value in options.items():
            solver.options[option_name] = option_value

        result = solver.solve(self.model)

        self.data['x_opt'] = [self.model.x[i].value for i in self.model.x]
        self.data['P_opt'] = self.data['x_opt'] * self.data['P']
        self.data['Q_opt'] = [self.Q[i] * pyo.value(self._el(i)) for i in self.model.x]

        status = str(result.solver.status)

        return {
            'message': str(result.solver.termination_condition),
            'status': status,
            'model': self.model,
            'data': self.data
        }


class PyomoLpOptimizationModel(OptimizationModel):

    def __init__(self, data):
        super().__init__(data, 'data_milp')

        self.N = len(self.data['plu_line_idx'])
        self.grid_size = self.data['grid_size'].values
        self.g_max = max(self.grid_size)
        self.plu_line_idx = self.data['plu_line_idx'].values
        self.n_plu = self.data['n_plu'].values
        self.P_idx = self.data['P_idx'].values
        self.Ps = np.vstack(self.data['Ps'].values)
        self.Qs = np.vstack(self.data['Qs'].values)

        # границы для индексов
        self.xs = self.data['xs'].values
        # Задаём объект модели pyomo
        self.model = pyo.ConcreteModel()

    def init_variables(self):
        # задаем бинарную метку для цены
        def init_fun(model, i, j):
            return 1 if self.P_idx[i] == j else 0

        self.model.x = pyo.Var(range(self.N), range(self.g_max), initialize=init_fun, domain=pyo.Binary)
        # одна единичка для каждого товара
        self.model.con_any_price = pyo.Constraint(pyo.Any)
        for i in range(self.N):
            self.model.con_any_price[i] = sum(self.model.x[i, j] for j in range(self.grid_size[i])) == 1

    def init_objective(self):
        objective = sum(sum(self.Ps * self.Qs * self.model.x))
        self.model.obj = pyo.Objective(expr=objective, sense=pyo.maximize)

    def add_con_mrg(self, m_min, m_max=None):
        con_mrg_expr = sum(sum((self.Ps - self.C.reshape(-1, 1)) * self.Qs * self.model.x)) >= m_min
        self.model.con_mrg = pyo.Constraint(expr=con_mrg_expr)

    def add_con_chg_cnt(self, nmax=10000):
        con_expr = sum(self.model.x[i, self.P_idx[i]] * self.n_plu[i]
                       for i in range(self.N) if self.P_idx[i] > 0) >= sum(self.n_plu) - nmax
        self.model.con_chg_cnt = pyo.Constraint(expr=con_expr)

    def solve(self, solver='cbc', options={}):
        solver = pyo.SolverFactory(solver, io_format='python', symbolic_solver_labels=False)
        for option_name, option_value in options.items():
            solver.options[option_name] = option_value
        result = solver.solve(self.model)
        x_sol = [[self.model.x[i, j].value for j in range(self.grid_size[i])] for i in range(self.N)]
        x_opt_idx = [np.argmax(x_sol[i]) for i in range(self.N)]
        x_opt = [self.xs[i][x_opt_idx[i]] for i in range(self.N)]
        P_opt = [self.Ps[i][x_opt_idx[i]] for i in range(self.N)]
        Q_opt = [self.Qs[i][x_opt_idx[i]] for i in range(self.N)]

        self.data['P_opt'] = P_opt
        self.data['Q_opt'] = Q_opt
        self.data['x_opt'] = x_opt
        return {
            'message': str(result.solver.termination_condition),
            'status': str(result.solver.status),
            'model': self.model,
            'data': self.data,
            'x_sol': x_sol,
            'opt_idx': x_opt_idx
        }


class CvxpyLpOptimizationModel(OptimizationModel):

    def __init__(self, data):
        super().__init__(data, 'data_milp')
        self.grid_size = self.data['grid_size'].values
        self.g_max = max(self.grid_size)
        self.plu_line_idx = self.data['plu_line_idx'].values
        self.n_plu = self.data['n_plu'].values
        self.P_idx = self.data['P_idx'].values
        self.Ps = np.array(self.data['Ps'].to_list())
        self.Qs = np.array(self.data['Qs'].to_list())
        self.C = self.data['C'].values.reshape(-1, 1)
        # границы для индексов
        self.xs = np.array(self.data['xs'].to_list())
        # Задаём объекты для формирования
        self.x = None
        self.obj = None
        self.constraints = {}
        self.x_mask = None

    def init_variables(self):
        self.x = cp.Variable(shape=(self.N, self.g_max), boolean=True)
        # должна быть хотя бы одна цена из диапазона
        # вспомогательная маска для упрощения матричных операций при формирований задачи
        mask_idx = np.repeat(np.arange(self.g_max), self.N).reshape(self.g_max, self.N).T
        mask = np.ones((self.N, self.g_max))
        mask[mask_idx > np.array(self.grid_size).reshape(-1, 1) - 1] = 0
        self.x_mask = mask
        con_any_price = cp.sum(cp.multiply(self.x, self.x_mask), axis=1) == 1
        self.constraints['var_any_price'] = con_any_price

    def init_objective(self):
        self.obj = cp.Maximize(cp.sum(cp.multiply(self.x, self.Ps * self.Qs)))

    def add_con_mrg(self, m_min):
        con_mrg = cp.sum(cp.multiply(self.x, (self.Ps - self.C) * self.Qs)) >= m_min
        self.constraints['con_mrg'] = con_mrg

    def add_con_chg_cnt(self, nmax=10000):
        con_chg_cnt = cp.sum(
            cp.multiply(self.x[np.arange(self.N), self.P_idx], self.n_plu)[self.P_idx > 0]
        ) >= sum(self.n_plu) - nmax
        self.constraints['con_chg_cnt'] = con_chg_cnt

    def solve(self, solver='ECOS_BB', options={}):
        problem = cp.Problem(self.obj, self.constraints.values())
        problem.solve(solver, **options)

        if self.x.value is None:
            return {
                'message': str(problem.solution.status),
                'status': str(problem.status),
                'model': problem,
                'data': self.data,
            }

        x_opt_idx = [np.argmax(self.x.value[i, :self.grid_size[i]]) for i in range(self.N)]
        x_opt = self.xs[np.arange(self.N), x_opt_idx]
        P_opt = self.Ps[np.arange(self.N), x_opt_idx]
        Q_opt = self.Qs[np.arange(self.N), x_opt_idx]
        self.data['P_opt'] = P_opt
        self.data['Q_opt'] = Q_opt
        self.data['x_opt'] = x_opt

        status = str(problem.status)
        status = 'ok' if status == 'optimal' else status

        return {
            'message': str(problem.solution.status),
            'status': status,
            'model': problem,
            'data': self.data,
            'opt_idx': x_opt_idx
        }
