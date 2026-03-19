import os
import re
from itertools import product
from typing import List, Optional, Dict, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp

from math import comb


INFINITY = 1e5
EPSILON = 1e-6



# =========================
# Ode class utils
# =========================
def get_orders(n_variables, order=2):
    '''
    tmp: get_orders(2) --> [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
    '''
    if n_variables == 1:
        orders = [[order]]
    else:
        orders = []
        if order == 0:
            orders.append([0] * n_variables)
        elif order == 1:
            for i in range(n_variables):
                orders.append([int(i==j) for j in range(n_variables)] )
        elif order >= 2:
            for first_order in range(order+1)[::-1]:
                rest_orders = get_orders(
                    n_variables=n_variables-1, order=order-first_order)
                for i in rest_orders:
                    orders.append([first_order] + i)
    return orders

def form_orders(n_variables=3, max_order=2):
    '''
    return [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
    '''
    orders = []
    for order in range(max_order+1):
        orders += get_orders(n_variables, order)
    return orders

def get_terms(variables='xyz', max_order=2, extra_term_dicts=None):
    '''
    ['', 'x', 'y', 'z', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    '''
    if extra_term_dicts is None:
        extra_term_dicts = {}
    dimension = len(variables)
    orders = form_orders(n_variables=dimension, max_order=max_order)
    terms = []
    for i_order in range(len(orders)):
        new_term_name = ''
        for i_variable in range(dimension):
            for _ in range(orders[i_order][i_variable]):
                new_term_name += variables[i_variable]

        terms.append(new_term_name)
    terms += [*extra_term_dicts.keys()]

    return terms

def get_basis(X, max_order=2, extra_term_dicts=None):
    if extra_term_dicts is None:
        extra_term_dicts = {}
    X = np.array(X)
    if len(X.shape) == 1:
        X = np.reshape(X, (1, -1))
    X_ = PolynomialFeatures(degree=max_order).fit_transform(X)
    X_ = np.concatenate([X_] + [f(X)
                                for f in extra_term_dicts.values()], axis=1)
    return X_



# =========================
# Ode class
# =========================
class ODE(object):
    def __init__(self, name='ode', dimension=3, max_order=2, coef=None, h=0.01,
                  extra_term_dicts=None, grid_limits=None, system_velocity=None):
        self.name = name
        self.dimension = dimension
        self.max_order = max_order
        self.variables = ['x', 'y', 'z']
            
        if extra_term_dicts is None:
            self.extra_term_dicts = {}
        else:
            self.extra_term_dicts = extra_term_dicts

        self.terms = get_terms(
            self.variables, self.max_order, self.extra_term_dicts)

        self.coef_shape = (self.dimension, len(self.terms))

        if coef is None:
            self.coef = np.zeros(self.coef_shape)
            self.coef_init = np.zeros(self.coef_shape)
        else:
            assert coef.shape[1] == self.coef_shape[1]
            self.coef = np.copy(coef)
            self.coef_init = np.copy(coef)

        self.h = h
        self._info = None

        self.grid_limits = grid_limits
        self.system_velocity = system_velocity


    def set_coef(self, eqn_index, term, value):
        if term in self.terms:
            self.coef_init[eqn_index, self.terms.index(term)] = value
            self.coef[eqn_index, self.terms.index(term)] = value
        elif term in ('1', ''):
            self.coef_init[eqn_index, 0] = value
            self.coef[eqn_index, 0] = value
        else:
            raise KeyError(f'Term {term} not found!')


    def d(self, points):
        points_ = get_basis(X=points, max_order=self.max_order, extra_term_dicts=self.extra_term_dicts)
        return points_ @ self.coef.T


    def R_K(self, point, h=None):
        if h is None:
            h = self.h
        point = np.array(point).reshape((1, -1))
        K1 = self.d(point)
        K2 = self.d(point + h / 2 * K1)
        K3 = self.d(point + h / 2 * K2)
        K4 = self.d(point + h * K3)
        return point + h / 6 * (K1 + 2*K2 + 2*K3 + K4)


    def reset_coef(self):
        self.coef = np.copy(self.coef_init)


    def exp(self, init_point=None, num_points=5000, omit_points=0, h=None, bounds=None, return_status=False, epsilon=EPSILON):
        # Set integration step
        if h is None:
            h = self.h

        if init_point is None:
            init_point = np.random.normal(size=(1, self.dimension))
            for _ in range(100):
                init_point = self.R_K(init_point)
        else:
            init_point = np.array(init_point).reshape((1, -1))

        points = [init_point]
        for i in range(1, num_points):
            points.append(self.R_K(points[-1], h=h))
            if np.sum(np.abs(points[-1]-points[-2])) < epsilon:
                print(f'Value Converged at point {i} at {np.round(points[-1][0], 3)}')
                status = f'Converged at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})'
                break
            if np.abs(points[-1]).max() > INFINITY:
                print(f'Value Exploded at point {i} on variable {self.variables[np.argmax(np.abs(points[-1]))]}!')
                status = f'Exploded on variable {self.variables[np.argmax(np.abs(points[-1]))]}!'
                break
            if not bounds is None:
                # bounds = [(0, 1)] * self.dimension
                if points[-1][0, 0] < bounds[0][0] or points[-1][0, 0] > bounds[0][1] or \
                    points[-1][0, 1] < bounds[1][0] or points[-1][0, 1] > bounds[1][1] or \
                    points[-1][0, 2] < bounds[2][0] or points[-1][0, 2] > bounds[2][1]:
                    print(f'Point {i} Out of Bounds!')
                    status = f'Out of Bounds!'
                    break
        else:
            status = 'Normal'
        points = np.concatenate(points, axis=0)
        if len(points) > omit_points:
            if return_status:
                return points[omit_points:], status
            else:
                return points[omit_points:]
        else:
            if return_status:
                return points, status
            else:
                return points


    def plot(self, points=None, init_point=None, num_points=5000, omit_points=0, h=None,
            subplot=111, pic_save_dir=None, plot_simple=False,
            figsize=(20, 5), use_time_axis=True):
        """
        use_time_axis : bool
            True  -> x axis corresponds to time t (t = step * h)
            False -> x axis corresponds to step   (0,1,2,...)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if points is None:
            points = self.exp(init_point=init_point, num_points=num_points,
                            omit_points=omit_points, h=h)
        points = np.asarray(points)
        n = points.shape[0]

        # integration step
        if h is None:
            h = self.h
        steps = np.arange(n)
        t = steps * h
        x_axis = t if use_time_axis else steps

        fig = plt.figure(figsize=figsize)
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)

        # ===== left: phase space trajectory =====
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        if plot_simple:
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])
            ax3d.grid(False)

        ax3d.plot(xs, ys, zs, color='#f79059')
        ax3d.set_xlabel(self.variables[0])
        ax3d.set_ylabel(self.variables[1])
        ax3d.set_zlabel(self.variables[2])

        # ===== right: integral curve =====
        for d in range(self.dimension):
            ax2d.plot(x_axis, points[:, d], label=self.variables[d])
        ax2d.set_xlabel('t' if use_time_axis else 'step')
        ax2d.grid(True, alpha=0.3)
        ax2d.legend()

        plt.tight_layout()

        if pic_save_dir:
            plt.savefig(pic_save_dir + '.png', dpi=450, bbox_inches='tight', pad_inches=0)


    def get_info(self, round_digits=3, epsilon=0.001):
        self._info = ''
        for i in range(self.dimension):
            self._info += f'd{self.variables[i]}/dt = ' 
            line = (' + '.join([(str(round(self.coef_init[i, j], round_digits)) + self.terms[j])
                                 for j in range(self.coef_shape[1]) if abs(self.coef_init[i, j]) > epsilon]))
            if line:
                self._info += line
            else:
                self._info += '0'
            self._info += '\n'
        self._info += f'name = {self.name}'
        self._info += '\n'
        self._info += f'grid_limits = {self.grid_limits}'
        self._info += '\n'
        self._info += f'system_velocity(avg) = {self.system_velocity}'
        return self._info


    def spatial_rescale(self, new_limits, name=None):
        if self.extra_term_dicts:
            raise NotImplementedError("affine_rescale_to polynominal")

        if self.grid_limits is None:
            raise ValueError("set self.grid_limits")

        old_limits = self.grid_limits
        old_lower = np.array([L[0] for L in old_limits], dtype=float)
        old_upper = np.array([L[1] for L in old_limits], dtype=float)
        new_lower = np.array([L[0] for L in new_limits], dtype=float)
        new_upper = np.array([L[1] for L in new_limits], dtype=float)

        old_span = old_upper - old_lower
        new_span = new_upper - new_lower
        if np.any(new_span == 0) or np.any(old_span == 0):
            raise ValueError("bound error")

        # D = diag(old_span / new_span)
        s = old_span / new_span
        b = old_lower - s * new_lower

        n_poly = _poly_term_count(self.dimension, self.max_order)
        A = _build_affine_basis_matrix(b=b, s=s, dimension=self.dimension, max_order=self.max_order)
        S_inv = np.diag(1.0 / s)
        # basis_x(x(y)) = basis_y(y) @ A  ==>  coef_new = D^{-1} @ coef_old @ A^T
        coef_new_poly = (S_inv @ self.coef[:, :n_poly]) @ A.T

        if self.coef.shape[1] != n_poly:
            raise NotImplementedError("special feature")

        new_coef = np.copy(self.coef)
        new_coef[:, :n_poly] = coef_new_poly

        new_name = name if name is not None else (self.name + "_affine")
        new_ode = ODE(name=new_name,
                      dimension=self.dimension,
                      max_order=self.max_order,
                      coef=new_coef,
                      h=self.h,
                      extra_term_dicts={},
                      grid_limits=[list(L) for L in new_limits]) 
        return new_ode


    def estimate_system_velocity(self, method='grid', n_per_axis=48, n_samples=20000, norm='l2', set_attr=True, random_state=0):
        if self.grid_limits is None:
            raise ValueError("estimate_system_velocity need grid_limits")

        lowers = np.array([a for a, b in self.grid_limits], dtype=float)
        uppers = np.array([b for a, b in self.grid_limits], dtype=float)

        if method == 'grid':
            axes = [np.linspace(lowers[i], uppers[i], n_per_axis) for i in range(self.dimension)]
            meshes = np.meshgrid(*axes, indexing='ij')
            X = np.stack([m.reshape(-1) for m in meshes], axis=1)
        elif method == 'random':
            rng = np.random.default_rng(random_state)
            U = rng.random((n_samples, self.dimension))
            X = lowers + U * (uppers - lowers)
        else:
            raise ValueError("method must be 'grid' or 'random'")

        V = self.d(X)  # (N,d)
        if norm == 'l2':
            speeds = np.linalg.norm(V, axis=1)
        elif norm == 'l1':
            speeds = np.linalg.norm(V, ord=1, axis=1)
        else:
            raise ValueError("norm must be 'l2' or 'l1'")

        avg_speed = float(np.mean(speeds))
        if set_attr:
            self.system_velocity = avg_speed
        return avg_speed


    def velocity_rescale(self, target_avg_speed, name=None, 
                            estimate_method='grid', n_per_axis=48, n_samples=20000, norm='l2', random_state=0):
        if target_avg_speed <= 0:
            raise ValueError("target_avg_speed should be positive")

        current = self.estimate_system_velocity(method=estimate_method, n_per_axis=n_per_axis,
                                                n_samples=n_samples, norm=norm, set_attr=False,
                                                random_state=random_state)
        if current == 0.0:
            raise ValueError("avg speed = 0")

        s = float(target_avg_speed / current)
        new_coef = s * self.coef

        new_name = name if name is not None else (self.name + "_velscaled")

        new_ode = ODE(name=new_name,
                      dimension=self.dimension,
                      max_order=self.max_order,
                      coef=new_coef,
                      h=self.h,
                      extra_term_dicts=self.extra_term_dicts,
                      grid_limits=self.grid_limits,
                      system_velocity=target_avg_speed)

        return new_ode



# =========================
# ode2vec and vec2ode
# =========================
def calc_grids(grid_limits, grid_shape):
    '''
    grid_shape = [48, 48, 48]
    grid_limits = [[0, 1], [0, 1], [0, 1]]
    grids = calc_grids(grid_limits=grid_limits, grid_shape=grid_shape)
    '''
    grids = [np.linspace(start=grid_limits[i][0], stop=grid_limits[i][1],
                    num=grid_shape[i]) for i in range(len(grid_shape))]
    return grids

def ode2vec(ode, grids):
    X = np.array([x for x in product(*grids)]).reshape((-1, len(grids)))
    vec = ode.d(X).reshape([len(grid) for grid in grids] + [len(grids)])
    return vec

def vec2ode(
    vec, grids, sparse_coef=None, max_order=2, extra_term_dicts=None,
    mode='linear', alpha=0.1, alphas=None, cv=5, fit_intercept=True,
    name='ode_hat', verbose=False, return_diagnostics=False, grid_limits=None
):
    """
    Supported: 'linear' | 'ridge' | 'lasso' | 'ridge_cv' | 'lasso_cv'
    - alphas: alpha grid for CV (use default logspace if None)
    - verbose: print per-dimension MSE/MAE/R2, penalty term, and alpha during fitting
    - return_diagnostics: return (ode, diagnostics); otherwise only return ode

    Notes:
    - Always drop the constant column from PolynomialFeatures; let the model fit the intercept (fit_intercept=True).
    - The sparse mask `sparse_coef` must have shape (d, n_features_full) — including the original bias column;
    in practice, subsetting is applied only to the features *after* removing the bias.
    """
    if extra_term_dicts is None:
        extra_term_dicts = {}

    from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    dimension = len(grids)
    Y = vec.reshape((-1, dimension))
    meshes = np.meshgrid(*grids, indexing='ij')
    X = np.stack([m.reshape(-1) for m in meshes], axis=1)

    X_full = get_basis(X=X, max_order=max_order, extra_term_dicts=extra_term_dicts)
    n_samples, n_features_full = X_full.shape

    feat_idx_all = np.arange(1, n_features_full)
    X_used_all = X_full[:, feat_idx_all]

    has_mask = sparse_coef is not None
    if has_mask:
        sparse_coef = np.asarray(sparse_coef)
        assert sparse_coef.shape == (dimension, n_features_full), \
            f"sparse_coef shape should be ({dimension}, {n_features_full})"
        sparse_mask_all = sparse_coef[:, 1:].astype(bool) 

    def make_estimator():
        if mode == 'linear':
            return LinearRegression(fit_intercept=fit_intercept)
        elif mode == 'ridge':
            return Ridge(alpha=alpha, fit_intercept=fit_intercept)
        elif mode == 'lasso':
            return Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
        elif mode == 'ridge_cv':
            grid = alphas if alphas is not None else np.logspace(-6, 2, 30)
            return RidgeCV(alphas=grid, fit_intercept=fit_intercept, store_cv_values=False)
        elif mode == 'lasso_cv':
            return LassoCV(alphas=alphas, cv=cv, fit_intercept=fit_intercept, max_iter=10000)
        else:
            raise ValueError("mode should be 'linear'|'ridge'|'lasso'|'ridge_cv'|'lasso_cv'")

    coef = np.zeros((dimension, n_features_full))

    diagnostics = {
        'per_dim': [],
        'mode': mode,
        'max_order': max_order,
        'n_samples': n_samples,
        'n_features_full': n_features_full,
        'used_feature_count_per_dim': [],
    }

    for i in range(dimension):
        if has_mask:
            mask_i = sparse_mask_all[i]
            if not np.any(mask_i):
                y_i = Y[:, i]
                y_mean = y_i.mean()
                coef[i, :] = 0.0
                if fit_intercept:
                    coef[i, 0] = y_mean
                mse = np.mean((y_i - y_mean)**2)
                mae = np.mean(np.abs(y_i - y_mean))
                r2 = 0.0 
                diag = {
                    'dim': i, 'alpha_used': None, 'mse': mse, 'mae': mae, 'r2': r2,
                    'penalty_L1': 0.0, 'penalty_L2': 0.0, 'n_features_used': 0
                }
                diagnostics['per_dim'].append(diag)
                diagnostics['used_feature_count_per_dim'].append(0)
                if verbose:
                    print(f"[dim {i}] intercept-only: MSE={mse:.4e}, MAE={mae:.4e}, R2={r2:.4f}")
                continue

            X_i = X_used_all[:, mask_i]
            n_used = int(mask_i.sum())
        else:
            X_i = X_used_all
            n_used = X_i.shape[1]

        y_i = Y[:, i]
        est = make_estimator()
        est.fit(X_i, y_i)

        y_pred = est.predict(X_i)
        mse = mean_squared_error(y_i, y_pred)
        mae = mean_absolute_error(y_i, y_pred)
        r2 = r2_score(y_i, y_pred)

        coef_vec = est.coef_
        penalty_L1 = float(np.abs(coef_vec).sum())
        penalty_L2 = float(np.dot(coef_vec, coef_vec))  # ||w||^2

        alpha_used = None
        if mode in ('ridge', 'lasso'):
            alpha_used = alpha
        elif mode == 'ridge_cv':
            alpha_used = float(est.alpha_)
        elif mode == 'lasso_cv':
            alpha_used = float(est.alpha_)

        coef[i, 1:][mask_i if has_mask else slice(None)] = coef_vec
        if fit_intercept:
            coef[i, 0] = est.intercept_

        diag = {
            'dim': i,
            'alpha_used': alpha_used,
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'penalty_L1': penalty_L1,
            'penalty_L2': penalty_L2,
            'n_features_used': int(n_used),
        }
        diagnostics['per_dim'].append(diag)
        diagnostics['used_feature_count_per_dim'].append(int(n_used))

        if verbose:
            pen = ""
            if mode.startswith('ridge'):
                pen = f", ||w||_2^2={penalty_L2:.3e}"
            elif mode.startswith('lasso'):
                pen = f", ||w||_1={penalty_L1:.3e}"
            print(f"[dim {i}] alpha={alpha_used}, MSE={mse:.4e}, MAE={mae:.4e}, R2={r2:.4f}{pen} (features used: {n_used})")

    ode = ODE(name=name, coef=coef, max_order=max_order,
              dimension=dimension, extra_term_dicts=extra_term_dicts, grid_limits=grid_limits)

    if return_diagnostics:
        return ode, diagnostics
    else:
        return ode



# =========================
# tensor2vec and vec2tensor
# =========================
def tensor2vec(image_tensor):
    # from tensor(1, channels, D, H, W) to ndarray(D, H, W, channels)
    return image_tensor.squeeze(0).permute(1, 2, 3, 0).numpy()

def vec2tensor(image_vec):
    # from ndarray(D, H, W, channels) to tensor(1, channels, D, H, W)
    return torch.tensor(image_vec, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)



# =========================
# spatial rescale
# =========================
def _poly_term_count(dimension, max_order):
    return len(form_orders(n_variables=dimension, max_order=max_order))

def _build_affine_basis_matrix(b, s, dimension, max_order):
    orders = form_orders(n_variables=dimension, max_order=max_order)  # list of multi-index
    n = len(orders)
    A = np.zeros((n, n), dtype=float)

    order_to_idx = {tuple(o): i for i, o in enumerate(orders)}

    for j, alpha in enumerate(orders):
        alpha = np.array(alpha, dtype=int)
        ranges = [range(a_i + 1) for a_i in alpha]
        for ks in product(*ranges):
            ks = np.array(ks, dtype=int) 
            coeff = 1.0
            for i in range(dimension):
                ai = int(alpha[i]); ki = int(ks[i])
                coeff *= comb(ai, ki) * (b[i] ** (ai - ki)) * (s[i] ** ki)
            i_y = order_to_idx[tuple(ks)]
            A[i_y, j] += coeff

    return A



# =========================
# ensemble rescaling
# =========================
def ensemble_rescaling(system,
                       target_grid_limits: list[list] = [[0,1],[0,1],[0,1]],
                       target_velocity: float = 5.0
                       ):
    system_unit_name_buffer = system.name + '_unit'
    system_unit = system.spatial_rescale(target_grid_limits, name=system_unit_name_buffer)

    system_unit_nv_name_buffer = system_unit_name_buffer + '_nv'
    system_unit.estimate_system_velocity(method='grid', n_per_axis=48)
    system_unit_nv = system_unit.velocity_rescale(target_avg_speed=target_velocity,
                                                  name=system_unit_nv_name_buffer,
                                                  estimate_method='grid',
                                                  n_per_axis=48
                                                  )
    
    return system_unit, system_unit_nv



# =========================
# homotopy methods
# =========================
def assert_same_basis(odeA, odeB):
    if odeA.dimension != odeB.dimension:
        raise ValueError(f"dimension mismatch: {odeA.dimension} vs {odeB.dimension}")
    if odeA.max_order != odeB.max_order:
        raise ValueError(f"max_order mismatch: {odeA.max_order} vs {odeB.max_order}")
    if odeA.terms != odeB.terms:
        raise ValueError("terms mismatch: ODEs do not share the same basis (terms).")
    if set(getattr(odeA, "extra_term_dicts", {}).keys()) != set(getattr(odeB, "extra_term_dicts", {}).keys()):
        raise ValueError("extra_term_dicts keys mismatch.")

def mix_odes_by_coef(
    odeA,
    odeB,
    a: float,
    b: float,
    name: Optional[str] = None,
    grid_limits=None,
    h: Optional[float] = None,
    system_velocity=None,
):
    assert_same_basis(odeA, odeB)

    coef_mix = a * np.asarray(odeA.coef, dtype=float) + b * np.asarray(odeB.coef, dtype=float)

    if name is None:
        name = f"mix__{odeA.name}__{odeB.name}__a{a:.4f}_b{b:.4f}"

    if grid_limits is None:
        grid_limits = getattr(odeA, "grid_limits", None)

    if h is None:
        h = float(getattr(odeA, "h", 0.01))

    extra_term_dicts = getattr(odeA, "extra_term_dicts", {})

    ode_mix = ODE(
        name=name,
        dimension=odeA.dimension,
        max_order=odeA.max_order,
        coef=coef_mix,
        h=h,
        extra_term_dicts=extra_term_dicts,
        grid_limits=grid_limits,
        system_velocity=system_velocity,
    )
    return ode_mix

def lerp_odes(odeA, odeB, alpha: float, **kwargs):
    """
    alpha in [0,1]: (1-alpha)*A + alpha*B
    """
    alpha = float(alpha)
    return mix_odes_by_coef(odeA, odeB, a=1.0-alpha, b=alpha, **kwargs)



# =========================
# CAE distance utilities
# =========================
@torch.no_grad()
def ode_to_tensor(
    ode,
    grid_shape=(48, 48, 48),
    grid_limits=((0, 1), (0, 1), (0, 1)),
    device=None,
    dtype=None,
):
    """
    ode -> tensor of shape (1, 3, D, H, W)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grids = calc_grids(grid_limits=grid_limits, grid_shape=grid_shape)
    vec = ode2vec(ode, grids)                      # (D,H,W,3)
    vec = vec.transpose(3, 0, 1, 2)                # (3,D,H,W)
    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # (1,3,D,H,W)

    if dtype is None:
        dtype = x.dtype

    x = x.to(device=device, dtype=dtype, non_blocking=True)
    return x

def gram_matrix_3d(x: torch.Tensor, normalize: str = "none", eps: float = 1e-6):
    """
    x: (B,C,D,H,W) or (C,D,H,W)
    return: (B,C,C)
    normalize:
      - "none": raw Gram
      - "zscore": per-channel z-score over spatial positions before Gram (correlation-like)
      - "l2": per-channel L2 normalize before Gram
    """
    if x.dim() == 4:
        x = x.unsqueeze(0)
    if x.dim() != 5:
        raise ValueError(f"Expected 4D/5D tensor, got {x.shape}")

    x = x.contiguous()
    B, C = x.shape[:2]
    Fm = x.view(B, C, -1)  # (B,C,N)

    if normalize == "zscore":
        mu = Fm.mean(dim=-1, keepdim=True)
        sd = Fm.std(dim=-1, keepdim=True).clamp_min(eps)
        Fm = (Fm - mu) / sd
    elif normalize == "l2":
        denom = torch.linalg.norm(Fm, dim=-1, keepdim=True).clamp_min(eps)
        Fm = Fm / denom
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'none'|'zscore'|'l2'")

    G = torch.bmm(Fm, Fm.transpose(1, 2)) / Fm.shape[-1]  # (B,C,C)
    return G

def layerwise_style_content(
    feats_a: List[torch.Tensor],
    feats_b: List[torch.Tensor],
    layers: Optional[Sequence[int]] = None,
    gram_normalize: str = "none",
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    feats_a, feats_b: list of feature maps each (B,C,D,H,W)
    returns dict with per-layer and total style/content distances
    """
    if layers is None:
        layers = list(range(len(feats_a)))

    style_terms = []
    content_terms = []
    out = {}

    for li in layers:
        Fa = feats_a[li]
        Fb = feats_b[li]
        # content: feature MSE
        c = F.mse_loss(Fa, Fb, reduction=reduction)
        # style: Gram MSE
        Ga = gram_matrix_3d(Fa, normalize=gram_normalize)
        Gb = gram_matrix_3d(Fb, normalize=gram_normalize)
        s = F.mse_loss(Ga, Gb, reduction=reduction)

        out[f"content_l{li}"] = c
        out[f"style_l{li}"] = s
        content_terms.append(c)
        style_terms.append(s)

    out["content_total"] = torch.stack(content_terms).mean()
    out["style_total"] = torch.stack(style_terms).mean()
    return out

@torch.no_grad()
def cae_extract(cae, x: torch.Tensor):
    """
    x: (B,3,D,H,W)
    return feats(list), z, recon
    """
    feats, z, xhat = cae(x)
    return feats, z, xhat

@torch.no_grad()
def cae_distance_between_systems(
    cae,
    ode_a,
    ode_b,
    grid_shape=(48, 48, 48),
    grid_limits=((0, 1), (0, 1), (0, 1)),
    layers: Optional[Sequence[int]] = None,
    gram_normalize: str = "none",
    device=None,
):
    """
    Compute content/style distance between two ODEs via CAE features.
    """
    if device is None:
        device = next(cae.parameters()).device
    dtype = next(cae.parameters()).dtype

    xa = ode_to_tensor(ode_a, grid_shape, grid_limits, device=device, dtype=dtype)
    xb = ode_to_tensor(ode_b, grid_shape, grid_limits, device=device, dtype=dtype)

    feats_a, _, _ = cae_extract(cae, xa)
    feats_b, _, _ = cae_extract(cae, xb)

    return layerwise_style_content(
        feats_a, feats_b,
        layers=layers,
        gram_normalize=gram_normalize,
        reduction="mean",
    )

@torch.no_grad()
def pairwise_cae_distance_matrices(
    cae,
    odes: List,
    grid_shape=(48, 48, 48),
    grid_limits=((0, 1), (0, 1), (0, 1)),
    layers: Optional[Sequence[int]] = None,
    gram_normalize: str = "none",
    device=None,
):
    """
    Return (content_matrix, style_matrix, names).
    Matrices are (N,N) with totals averaged across selected layers.
    """
    if device is None:
        device = next(cae.parameters()).device
    dtype = next(cae.parameters()).dtype

    # build batch
    xs = []
    names = []
    for ode in odes:
        xs.append(ode_to_tensor(ode, grid_shape, grid_limits, device=device, dtype=dtype))
        names.append(getattr(ode, "name", "ode"))
    X = torch.cat(xs, dim=0)  # (N,3,D,H,W)

    feats, _, _ = cae_extract(cae, X)

    N = X.shape[0]
    Cmat = torch.zeros((N, N), device=device)
    Smat = torch.zeros((N, N), device=device)

    if layers is None:
        layers = list(range(len(feats)))

    # Precompute Gram for each layer to speed up
    grams = {}
    for li in layers:
        grams[li] = gram_matrix_3d(feats[li], normalize=gram_normalize)  # (N,C,C)

    # Compute pairwise
    for i in range(N):
        for j in range(i, N):
            c_terms = []
            s_terms = []
            for li in layers:
                c_terms.append(F.mse_loss(feats[li][i], feats[li][j], reduction="mean"))
                s_terms.append(F.mse_loss(grams[li][i], grams[li][j], reduction="mean"))
            c = torch.stack(c_terms).mean()
            s = torch.stack(s_terms).mean()
            Cmat[i, j] = Cmat[j, i] = c
            Smat[i, j] = Smat[j, i] = s

    return Cmat.detach().cpu().numpy(), Smat.detach().cpu().numpy(), names



# =========================
# System-level linear conjugacy transforms (exact for max_order=2, no extra terms)
# =========================
def _basis_T_for_linear_map_degree2(M: np.ndarray) -> np.ndarray:
    """
    Build T (10x10) such that:
      basis_x(x) with x = M y  equals  basis_y(y) @ T
    basis order: [1, y1, y2, y3, y1^2, y1y2, y1y3, y2^2, y2y3, y3^2]
    """
    if M.shape != (3, 3):
        raise ValueError("M must be 3x3")
    T = np.zeros((10, 10), dtype=float)

    # helper row indices for y monomials
    # 0:1, 1:y1,2:y2,3:y3, 4:y1^2,5:y1y2,6:y1y3,7:y2^2,8:y2y3,9:y3^2
    def idx_y2(j):  # y_j^2
        return [4, 7, 9][j]
    def idx_yjk(j, k):  # y_j y_k with j<k
        if (j, k) == (0, 1): return 5
        if (j, k) == (0, 2): return 6
        if (j, k) == (1, 2): return 8
        raise ValueError("bad pair")

    # constant
    T[0, 0] = 1.0

    # linear terms: x_i = sum_j M[i,j] y_j
    # x1 column = 1, x2 col=2, x3 col=3
    for i in range(3):
        col_xi = 1 + i
        for j in range(3):
            T[1 + j, col_xi] = M[i, j]

    # quadratic terms columns:
    # x1^2 col=4, x1x2 col=5, x1x3 col=6, x2^2 col=7, x2x3 col=8, x3^2 col=9
    # x_i^2
    for i in range(3):
        col = [4, 7, 9][i]
        # y_j^2 terms
        for j in range(3):
            T[idx_y2(j), col] += M[i, j] * M[i, j]
        # y_j y_k terms
        for j in range(3):
            for k in range(j + 1, 3):
                T[idx_yjk(j, k), col] += 2.0 * M[i, j] * M[i, k]

    # x_i x_k (i<k)
    pairs = [(0, 1, 5), (0, 2, 6), (1, 2, 8)]  # (i,k,col)
    for i, k, col in pairs:
        # y_j^2 terms
        for j in range(3):
            T[idx_y2(j), col] += M[i, j] * M[k, j]
        # y_j y_l terms
        for j in range(3):
            for l in range(j + 1, 3):
                T[idx_yjk(j, l), col] += (M[i, j] * M[k, l] + M[i, l] * M[k, j])

    return T

def _transform_grid_limits_bbox(grid_limits, A: np.ndarray):
    """
    Optional: transform axis-aligned limits by pushing 8 corners through y=Ax,
    then take per-axis min/max as new bbox.
    """
    lows = np.array([a for a, b in grid_limits], dtype=float)
    ups  = np.array([b for a, b in grid_limits], dtype=float)
    corners = np.array(list(product(*zip(lows, ups))), dtype=float)  # (8,3)
    yc = corners @ A.T
    new_limits = []
    for i in range(3):
        new_limits.append([float(yc[:, i].min()), float(yc[:, i].max())])
    return new_limits

def conjugate_linear_degree2(
    ode,
    A: np.ndarray,
    name: Optional[str] = None,
    transform_limits: str = "keep",  # "keep" | "bbox"
):
    """
    Exact linear conjugacy for polynomial vector fields up to degree 2 (no extra terms):
      y = A x
      f_new(y) = A f_old(A^{-1} y)

    Returns a NEW ODE with transformed coefficients.

    Caveat on grid_limits:
      - "keep": keep ode.grid_limits as-is (sampling domain in y-space unchanged)
      - "bbox": set new grid_limits to bbox of y=Ax applied to original x-box
    """
    if getattr(ode, "max_order", None) != 2:
        raise NotImplementedError("Only supports max_order=2 for exact transform.")
    if getattr(ode, "extra_term_dicts", None):
        raise NotImplementedError("extra_term_dicts must be empty for exact transform.")

    A = np.asarray(A, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("A must be 3x3")
    if np.linalg.det(A) == 0:
        raise ValueError("A must be invertible")

    M = np.linalg.inv(A)  # x = M y
    T = _basis_T_for_linear_map_degree2(M)  # basis_x(M y) = basis_y(y) @ T

    coef_old = np.asarray(ode.coef, dtype=float)  # (3,10)
    coef_new = A @ coef_old @ T.T                 # (3,10)

    new_name = name if name is not None else (ode.name + "_conj")
    new_limits = ode.grid_limits
    if (transform_limits == "bbox") and (ode.grid_limits is not None):
        new_limits = _transform_grid_limits_bbox(ode.grid_limits, A)

    return ODE(
        name=new_name,
        dimension=ode.dimension,
        max_order=ode.max_order,
        coef=coef_new,
        h=ode.h,
        extra_term_dicts={},  # enforced
        grid_limits=new_limits,
        system_velocity=getattr(ode, "system_velocity", None),
    )

def time_reverse_ode(ode, name: Optional[str] = None):
    """
    Time reversal for vector field sampling: f_new(x) = -f(x)
    """
    new_name = name if name is not None else (ode.name + "_timerev")
    return ODE(
        name=new_name,
        dimension=ode.dimension,
        max_order=ode.max_order,
        coef=-np.asarray(ode.coef),
        h=ode.h,
        extra_term_dicts=getattr(ode, "extra_term_dicts", {}),
        grid_limits=getattr(ode, "grid_limits", None),
        system_velocity=getattr(ode, "system_velocity", None),
    )

def permute_sign_conjugacy(ode, perm=(0, 1, 2), signs=(1, 1, 1), name=None, transform_limits="keep"):
    """
    y = A x where A = P * diag(signs), with P being permutation matrix on coordinates.
    This is a physically meaningful re-labeling + reflection.
    """
    perm = tuple(perm)
    signs = np.array(signs, dtype=float)
    P = np.zeros((3, 3), dtype=float)
    for i, pi in enumerate(perm):
        P[i, pi] = 1.0
    A = P @ np.diag(signs)
    new_name = name if name is not None else (ode.name + f"_perm{perm}_sgn{tuple(signs.astype(int))}")
    return conjugate_linear_degree2(ode, A=A, name=new_name, transform_limits=transform_limits)

def rotate_conjugacy(ode, R: np.ndarray, name=None, transform_limits="keep"):
    """
    y = R x, f_new(y) = R f_old(R^T y).  (R should be orthogonal ideally)
    """
    new_name = name if name is not None else (ode.name + "_rot")
    return conjugate_linear_degree2(ode, A=np.asarray(R, dtype=float), name=new_name, transform_limits=transform_limits)
