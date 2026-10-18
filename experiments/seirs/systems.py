import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as npl

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import os
import time

from mpl_toolkits.mplot3d import Axes3D

from experiments.seirs.fields import ODE as PolynomialODE, R

INFINITY = 1e3
EPSILON = 1e-6


class ODE(PolynomialODE):
    """System-bank integration keeps its original 1e3 divergence threshold."""

    integration_limit = INFINITY


init_point = np.random.normal(size=(1, 3))
# init_point = np.array([-5.76,  2.27,  32.82])

# lorenz_tiny
lorenz_tiny = ODE(name="lorenz_tiny", dimension=3, max_order=2)
# dx
lorenz_tiny.set_coef(0, "x", -10)
lorenz_tiny.set_coef(0, "y", 10)
# dy
lorenz_tiny.set_coef(1, "", -13.5)
lorenz_tiny.set_coef(1, "x", 28)
lorenz_tiny.set_coef(1, "y", -1)
lorenz_tiny.set_coef(1, "z", 25)
lorenz_tiny.set_coef(1, "xz", -50)
# dz
lorenz_tiny.set_coef(2, "", 12.5)
lorenz_tiny.set_coef(2, "x", -25)
lorenz_tiny.set_coef(2, "y", -25)
lorenz_tiny.set_coef(2, "z", -8 / 3)
lorenz_tiny.set_coef(2, "xy", 50)
# turb
turb_mode = "rate"
turb_args = np.zeros(lorenz_tiny.coef_shape + (2,))
turb_args[:, :, 0] = np.copy(lorenz_tiny.coef_init)
turb_args[:, :, 1] = np.ones(lorenz_tiny.coef_shape) * 0.1
lorenz_tiny.set_turb(turb_mode, turb_args)

# lorenz
lorenz = ODE(name="lorenz", dimension=3, max_order=2)
lorenz._sigma, lorenz._b, lorenz._r = 10.0, 8 / 3, 28.0
# dx = -\sigma*x+\sigma*y
lorenz.set_coef(0, "x", -lorenz._sigma)
lorenz.set_coef(0, "y", lorenz._sigma)
# dy = -xz + rx - y
lorenz.set_coef(1, "xz", -1)
lorenz.set_coef(1, "x", lorenz._r)
lorenz.set_coef(1, "y", -1)
# dz = xy - bz
lorenz.set_coef(2, "xy", 1)
lorenz.set_coef(2, "z", -lorenz._b)
# turb
turb_mode = "rate"
turb_args = np.zeros(lorenz.coef_shape + (2,))
turb_args[:, :, 0] = np.copy(lorenz.coef_init)
turb_args[:, :, 1] = np.ones(lorenz.coef_shape) * 0.1
lorenz.set_turb(turb_mode, turb_args)

# chen
chen = ODE(name="chen")
chen._a, chen._b, chen._c = 35.0, 3.0, 28.0
# dx = -ax+ay
chen.set_coef(0, "x", -chen._a)
chen.set_coef(0, "y", chen._a)
# dy = -xz + (c-a)x + cy
chen.set_coef(1, "xz", -1)
chen.set_coef(1, "x", chen._c - chen._a)
chen.set_coef(1, "y", chen._c)
# dz = xy - bz
chen.set_coef(2, "xy", 1)
chen.set_coef(2, "z", -chen._b)
# turb
turb_mode = "rate"
turb_args = np.zeros(chen.coef_shape + (2,))
turb_args[:, :, 0] = np.copy(chen.coef_init)
turb_args[:, :, 1] = np.ones(chen.coef_shape) * 0.1
chen.set_turb(turb_mode, turb_args)

# chen_tiny
chen_tiny = ODE(name="chen_tiny")
# dx
chen_tiny.set_coef(0, "x", -35)
chen_tiny.set_coef(0, "y", 43.75)
chen_tiny.set_coef(0, "", -4.375)
# dy
chen_tiny.set_coef(1, "x", -5.6)
chen_tiny.set_coef(1, "y", 28)
chen_tiny.set_coef(1, "z", 20)
chen_tiny.set_coef(1, "xz", -40)
chen_tiny.set_coef(1, "", -11.2)
# dz
chen_tiny.set_coef(2, "x", -80)
chen_tiny.set_coef(2, "y", -80)
chen_tiny.set_coef(2, "z", -3)
chen_tiny.set_coef(2, "xy", 160)
chen_tiny.set_coef(2, "", 40)
# turb
turb_mode = "rate"
turb_args = np.zeros(chen_tiny.coef_shape + (2,))
turb_args[:, :, 0] = np.copy(chen_tiny.coef_init)
turb_args[:, :, 1] = np.ones(chen_tiny.coef_shape) * 0.1
chen_tiny.set_turb(turb_mode, turb_args)

# lv
lv = ODE(name="lv")
lv._a, lv._b, lv._c, lv._u = 36, 3, 20, -15.15
# dx = -ax + ay
lv.set_coef(0, "x", -lv._a)
lv.set_coef(0, "y", lv._a)
# dy = - xz + x + cy + u
lv.set_coef(1, "xz", -1)
lv.set_coef(1, "x", 1)
lv.set_coef(1, "y", lv._c)
lv.set_coef(1, "", lv._u)
# dz = xy - bz
lv.set_coef(2, "xy", 1)
lv.set_coef(2, "z", -lv._b)
# turb
turb_mode = "rate"
turb_args = np.zeros(lv.coef_shape + (2,))
turb_args[:, :, 0] = np.copy(lv.coef_init)
turb_args[:, :, 1] = np.ones(lv.coef_shape) * 0.1
lv.set_turb(turb_mode, turb_args)

# lv_tiny
lv_tiny = ODE(name="lv_tiny")
# dx
lv_tiny.set_coef(0, "x", -36)
lv_tiny.set_coef(0, "y", 36)
# dy
lv_tiny.set_coef(1, "x", 1)
lv_tiny.set_coef(1, "y", 20)
lv_tiny.set_coef(1, "z", 25)
lv_tiny.set_coef(1, "", -10.803)
lv_tiny.set_coef(1, "xz", -50)
# dz
lv_tiny.set_coef(2, "x", -25)
lv_tiny.set_coef(2, "y", -25)
lv_tiny.set_coef(2, "z", -3)
lv_tiny.set_coef(2, "", 12.5)
lv_tiny.set_coef(2, "xy", 50)
# turb
turb_mode = "rate"
turb_args = np.zeros(lv_tiny.coef_shape + (2,))
turb_args[:, :, 0] = np.copy(lv_tiny.coef_init)
turb_args[:, :, 1] = np.ones(lv_tiny.coef_shape) * 0.1
lv_tiny.set_turb(turb_mode, turb_args)

"""
Def ode(seirs)
"""
seirs = ODE(
    name="seirs", variables=["S", "E", "I"], max_order=2, extra_term_dicts={"R": R}
)

seirs._beta = 0.9
seirs._sigma = 0.1
seirs._gamma = 0.2
seirs._xi = 0.15
seirs._N = 1

seirs.set_coef(0, "SI", -seirs._beta / seirs._N)
seirs.set_coef(0, "R", seirs._xi)

seirs.set_coef(1, "E", -seirs._sigma)
seirs.set_coef(1, "SI", seirs._beta / seirs._N)

seirs.set_coef(2, "I", -seirs._gamma)
seirs.set_coef(2, "E", seirs._sigma)

turb_mode = "uniform"
turb_args = np.zeros(seirs.coef_shape + (2,))
for i in range(seirs.coef_shape[0]):
    for j in range(seirs.coef_shape[1]):
        if np.abs(seirs.coef_init[i, j]) > EPSILON:
            # turb_args[i, j, 0] = 0.0
            if seirs.coef_init[i, j] > 0:
                turb_args[i, j, 1] = 1.0
            elif seirs.coef_init[i, j] < 0:
                turb_args[i, j, 1] = -1.0

seirs.set_turb(turb_mode, turb_args)

CXC = 0.4
CYC = 2.009
CXP = 0.08
CYP = 2.876
CR0 = 0.16129
CC0 = 0.5
CK = 0.99


def R_(X):
    R, C, P = X[:, [0]], X[:, [1]], X[:, [2]]
    return R / (R + CR0 + EPSILON)


def C_(X):
    R, C, P = X[:, [0]], X[:, [1]], X[:, [2]]
    return C / (C + CC0 + EPSILON)


def CR_(X):
    R, C, P = X[:, [0]], X[:, [1]], X[:, [2]]
    return C * R_(X)


def PC_(X):
    R, C, P = X[:, [0]], X[:, [1]], X[:, [2]]
    return P * C_(X)


rcp_extra_term_dicts = {"CR_": CR_, "PC_": PC_}
rcp = ODE(name="rcp", variables=["R", "C", "P"], extra_term_dicts=rcp_extra_term_dicts)

rcp.set_coef(0, "R", 1)
rcp.set_coef(0, "RR", -1 / CK)
rcp.set_coef(0, "CR_", -CXC * CYC)

rcp.set_coef(1, "CR_", CXC * CYC)
rcp.set_coef(1, "C", -CXC)
rcp.set_coef(1, "PC_", -CXP * CYP)

rcp.set_coef(2, "PC_", CXP * CYP)
rcp.set_coef(2, "P", -CXP)

turb_mode = "rate"
turb_args = np.zeros(rcp.coef_shape + (2,))
turb_args[:, :, 0] = np.copy(rcp.coef_init)
turb_args[:, :, 1] = np.ones(rcp.coef_shape) * 0.1

rcp.set_turb(turb_mode, turb_args)

# ode_list = [lorenz, chen, lv]
ode_list = []

ode_list.append(lorenz_tiny)
ode_list.append(chen_tiny)
ode_list.append(lv_tiny)
ode_list.append(rcp)
ode_list.append(seirs)

# '''
# Normalize ode to target grid: Create ode_shifted_list = [lorenz_shifted, chen_shifted, lv_shifted, seris, rossler_shifted, prs]
# '''
# grid_limits = [[0, 1], [0, 1], [0, 1]]
# grid_limits_origin = [[-25, 25], [-25, 25], [0, 50]]
# grid_shape = (48, 48, 48)

# ode_shifted_list = []
# for i_ode in [0, 1, 2]:
#     ode_shifted = ode_change_grid(ode_origin=ode_list[i_ode], grid_limits_origin=grid_limits_origin, grid_limits_target=grid_limits, grid_shape=grid_shape)
#     ode_shifted.name += '_shifted'
#     ode_shifted_list.append(ode_shifted)

# ode_shifted_list.append(rcp)
