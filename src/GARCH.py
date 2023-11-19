import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as np, jax.random as rnd
import jax.flatten_util
import equinox as eqx
from scipy.optimize import minimize

def lag(series: np.ndarray, lag: int=1):
    assert lag > 0
    shape = (lag, *series.shape[1:])
    return np.concatenate([np.zeros(shape, series.dtype), series[:-lag, ...]])

def npdf(x, mu, var):
    return np.exp(-0.5 * (x - mu)**2 / var) / np.sqrt(2 * np.pi * var)

def loss_normal(var_pred, series):
    logliks = -np.log(var_pred) - series**2 / var_pred
    return -logliks.mean()

def loss_mix(vars_pred, series, weights):
    # vars_pred: (T, K)
    # series: (T, 1)
    # weights: (K,)
    pdf_exponents = (
        - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(vars_pred) - 0.5 * series**2 / vars_pred
    ) # (T, K)
    logpdfs = jax.scipy.special.logsumexp(pdf_exponents, axis=1, b=weights)
    return -logpdfs.mean()

class GARCHCell(eqx.Module):
    bias: np.ndarray
    Wx: np.ndarray
    Wh: np.ndarray
    def __init__(self, *, key: rnd.PRNGKey):
        k1, k2 = rnd.split(key)
        self.bias = rnd.uniform(k1, (1, ))
        _, a, b = rnd.dirichlet(k2, np.ones(3))
        self.Wx, self.Wh = np.array([a]), np.array([b])

    def __call__(self, var_prev: np.ndarray, obs: np.ndarray):
        var = self.bias + self.Wx * obs + self.Wh * var_prev
        return var, var

class GARCHCellReLU(GARCHCell):
    def __init__(self, *, key: rnd.PRNGKey):
        self.bias, self.Wx, self.Wh = rnd.uniform(key, (3, 1))

    def __call__(self, var_prev: np.ndarray, obs: np.ndarray):
        var = jax.nn.relu(self.bias + self.Wx * obs + self.Wh * var_prev) + 1e-6
        return var, var

class GARCH(eqx.Module):
    cell: GARCHCell | GARCHCellReLU
    var0: np.ndarray = eqx.field(static=True)
    
    def __init__(self, var0: float, cell_cls=GARCHCell, *, key: rnd.PRNGKey):
        self.cell = cell_cls(key=key)
        self.var0 = np.array([var0])

    def __call__(self, series: np.ndarray):
        # `series` indexed by t=[0,1,2, ..., T]
        _, hist = jax.lax.scan(self.cell.__call__, self.var0, series**2)
        return hist # values for t=[1, 2, ..., T, T+1]

    def condvar(self, series):
        variances = self(series)
        return variances.flatten()

    def predict(self, series: np.ndarray):
        return self(series).flatten()[-1]
        
    def fit(self, series: np.ndarray, tol=1e-6):
        assert series.ndim == 2
        params0, rebuild = jax.flatten_util.ravel_pytree(self)
        @jax.jit
        def loss(params: np.ndarray):
            model = rebuild(params)
            vars_pred = model(series)
            return loss_normal(vars_pred[:-1, 0], series[1:, 0])
        @jax.jit
        def constr(params: np.ndarray):
            # a + b < 1 => 1 - (a + b) > 0
            model: GARCH = rebuild(params)
            return 1 - (model.cell.Wh[0] + model.cell.Wx[0])

        if isinstance(self.cell, GARCHCell):
            bounds = [
                (0, None), (0, None), (0, None), # GARCH params
            ]
            constraints = [{'type': 'ineq', 'fun': constr}]
        else:
            bounds = [(None, None), (None, None), (-1, 1)]
            constraints = None
        
        sol = minimize(
            loss, params0, method='SLSQP',
            bounds=bounds, constraints=constraints, tol=tol
        )
        N = series.shape[0]
        lnL = N * -loss(sol.x)
        aic = -2 * lnL + 2 * len(sol.x)
        bic = -2 * lnL + np.log(len(series)) * len(sol.x)
        return rebuild(sol.x), sol, {'aic': aic.item(), 'bic': bic.item()}

    def fit_predict(self, series: np.ndarray, tol=1e-6):
        model, *_ = self.fit(series, tol)
        return model, model.predict(series)

class MixGARCHCellReLU(eqx.Module):
    bias: np.ndarray
    Wx: np.ndarray
    Wh: np.ndarray
    def __init__(self, ncomp: int, n_in: int=1, *, key: rnd.PRNGKey):
        k1, k2, k3 = rnd.split(key, 3)
        init_bias = jax.nn.initializers.uniform()
        init_Wx = jax.nn.initializers.uniform()
        init_Wh = jax.nn.initializers.uniform(0.9)
        
        self.bias = init_bias(k1, (ncomp, ))
        self.Wx = init_Wx(k2, (ncomp, n_in))
        self.Wh = init_Wh(k3, (ncomp, ))

    def __call__(self, vars_prev: np.ndarray, obs: np.ndarray):
        vars = jax.nn.relu(self.bias + self.Wx @ obs + self.Wh * vars_prev) + 1e-6
        return vars, vars

class MixGARCH(eqx.Module):
    cell: MixGARCHCellReLU
    weights: np.ndarray
    vars0: np.ndarray = eqx.field(static=True)
    
    def __init__(self, vars0: np.ndarray, n_in: int=1, cell_cls=MixGARCHCellReLU, *, key: rnd.PRNGKey):
        self.vars0 = np.array(vars0)
        K = len(self.vars0)
        self.cell = cell_cls(K, n_in, key=key)
        self.weights = np.ones(K) / K

    def __call__(self, series: np.ndarray):
        """
        `series` must be (T, 1)
        Indexed by t=[0,1,2, ..., T]
        """
        _, hist = jax.lax.scan(self.cell.__call__, self.vars0, series**2)
        return hist # values for t=[1, 2, ..., T, T+1]

    def condvar(self, series):
        variances = self(series)
        return variances @ self.weights

    def predict(self, series: np.ndarray):
        return self.condvar(series).flatten()[-1]
        
    def fit(self, series: np.ndarray, tol=1e-6):
        """
        `series[:, 0]` must contain most recent lags of log-returns!
        """
        assert series.ndim == 2
        params0, rebuild = jax.flatten_util.ravel_pytree(self)
        @jax.jit
        def loss(params: np.ndarray):
            model = rebuild(params)
            vars_pred = model(series) # shape (T, K)
            return loss_mix(vars_pred[:-1, :], series[1:, :1], model.weights)
        @jax.jit
        def constr(params: np.ndarray):
            # a + b < 1 => 1 - (a + b) > 0
            model: MixGARCH = rebuild(params)
            return model.weights.sum() - 1 # == 0

        K = len(self.weights)
        sol = minimize(
            loss, params0, method='SLSQP',
            bounds=[(None, None)] * K + [(None, None)] * (K * series.shape[1]) + [(-1, 1)] * K + [(0, None)] * K, # bias, Wx, Wh, weights
            constraints=[{'type': 'eq', 'fun': constr}], tol=tol
        )

        N = series.shape[0]
        lnL = N * -loss(sol.x)
        aic = -2 * lnL + 2 * len(sol.x)
        bic = -2 * lnL + np.log(len(series)) * len(sol.x)
        return rebuild(sol.x), sol, {'aic': aic.item(), 'bic': bic.item()}

    def fit_predict(self, series: np.ndarray, tol=1e-6):
        model, *_ = self.fit(series, tol)
        return model, model.predict(series)
