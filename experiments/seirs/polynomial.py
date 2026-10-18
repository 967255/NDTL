"""Sparse polynomial reconstruction of the SEIRS vector fields."""

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge

BASIS_NAMES = ("1", "S", "E", "I", "S^2", "S*E", "S*I", "E^2", "E*I", "I^2")


def polynomial_basis():
    functions = (
        lambda s, e, i: np.ones_like(s),
        lambda s, e, i: s,
        lambda s, e, i: e,
        lambda s, e, i: i,
        lambda s, e, i: s**2,
        lambda s, e, i: s * e,
        lambda s, e, i: s * i,
        lambda s, e, i: e**2,
        lambda s, e, i: e * i,
        lambda s, e, i: i**2,
    )
    return {component: list(zip(functions, BASIS_NAMES)) for component in "xyz"}


class PolynomialField:

    def __init__(self, grid_limits, grid_shape, basis_funcs):
        """Initialize the fitting model with grid bounds, grid shape, and per-component basis functions."""
        self.grid_limits = grid_limits
        self.grid_shape = grid_shape
        self.basis_funcs = basis_funcs
        self.coeffs = {}
        self._generate_grid()

    def _generate_grid(self):
        self.grids = [
            np.linspace(lim[0], lim[1], num)
            for (lim, num) in zip(self.grid_limits, self.grid_shape)
        ]
        (self.xg, self.yg, self.zg) = self.grids
        (x, y, z) = np.meshgrid(self.xg, self.yg, self.zg, indexing="ij")
        self.x = x
        self.y = y
        self.z = z
        self.N = x.size
        self.x_flat = x.flatten()
        self.y_flat = y.flatten()
        self.z_flat = z.flatten()

    def _build_design_matrix_component(self, comp):
        """Evaluate the basis functions for one vector-field component on the flattened grid."""
        funcs = self.basis_funcs[comp]
        X = np.column_stack(
            [f(self.x_flat, self.y_flat, self.z_flat) for (f, name) in funcs]
        )
        return X

    def vector_field(self, t, state):
        """Vector field."""
        (x_val, y_val, z_val) = state
        basis_x = np.array(
            [f(x_val, y_val, z_val) for (f, name) in self.basis_funcs["x"]]
        )
        basis_y = np.array(
            [f(x_val, y_val, z_val) for (f, name) in self.basis_funcs["y"]]
        )
        basis_z = np.array(
            [f(x_val, y_val, z_val) for (f, name) in self.basis_funcs["z"]]
        )
        fx = np.dot(self.coeffs["x"], basis_x)
        fy = np.dot(self.coeffs["y"], basis_y)
        fz = np.dot(self.coeffs["z"], basis_z)
        return [fx, fy, fz]

    def fit(self, vectors, regression="lasso", alpha=0.005):
        """Fit three independent component regressions without an intercept."""
        estimators = {
            "linear": lambda: LinearRegression(fit_intercept=False),
            "lasso": lambda: Lasso(alpha=alpha, fit_intercept=False, max_iter=10000),
            "ridge": lambda: Ridge(alpha=alpha, fit_intercept=False),
        }
        if regression not in estimators:
            raise ValueError("regression must be linear, lasso, or ridge")
        if vectors.shape != (*self.grid_shape, 3):
            raise ValueError(f"expected {(*self.grid_shape, 3)}, got {vectors.shape}")
        for index, component in enumerate("xyz"):
            design = self._build_design_matrix_component(component)
            estimator = estimators[regression]().fit(
                design, vectors[..., index].flatten()
            )
            self.coeffs[component] = estimator.coef_
        return self

    @property
    def reproduction_ratio(self):
        """Return the fitted S*I coefficient in dE/dt divided by -I in dI/dt."""
        return self.coeffs["y"][6] / (-self.coeffs["z"][3])

    def coefficient_matrix(self):
        return np.array([self.coeffs[component] for component in "xyz"])


def load_sweep(path):
    """Load generated fields of shape (weights, 1, 48, 48, 48, 3)."""
    fields = np.load(path, mmap_mode="r", allow_pickle=False)
    if fields.ndim != 6 or fields.shape[1:] != (1, 48, 48, 48, 3):
        raise ValueError(f"expected (weights, 1, 48, 48, 48, 3), got {fields.shape}")
    return fields


def fit_field(vectors, alpha=0.005):
    return PolynomialField([[0, 1]] * 3, vectors.shape[:3], polynomial_basis()).fit(
        vectors, alpha=alpha
    )
