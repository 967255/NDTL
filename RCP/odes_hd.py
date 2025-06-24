import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as npl

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import os
import time

from mpl_toolkits.mplot3d import Axes3D

from utils_ode import *

INFINITY = 1e3
EPSILON = 1e-6


def get_orders(num_variables, order):
    if num_variables == 1:
        orders = [[order]]
    else:
        orders = []
        if order == 0:
            orders.append([0] * num_variables)
        elif order == 1:
            for i in range(num_variables):
                temp = [0] * num_variables
                temp[i] = 1
                orders.append(temp)
        elif order >= 2:
            for first_order in range(order+1)[::-1]:
                rest_orders = get_orders(
                    num_variables=num_variables-1, order=order-first_order)
                # print(rest_orders)
                for i in rest_orders:
                    orders.append([first_order] + i)
    return orders


def form_orders(num_variables=3, max_order=2):
    orders = []
    for order in range(max_order+1):
        orders += get_orders(num_variables, order)
    return orders


def get_terms(variables, max_order=2, extra_term_dicts={}):
    dimension = len(variables)
    orders = form_orders(num_variables=dimension, max_order=max_order)
    terms = []
    for i_order in range(len(orders)):
        new_term_name = ''
        for i_variable in range(dimension):
            for _ in range(orders[i_order][i_variable]):
                new_term_name += variables[i_variable]

        terms.append(new_term_name)
    terms += [*extra_term_dicts.keys()]

    return terms


def get_basis(X, max_order=2, extra_term_dicts={}):
    X = np.array(X)
    if len(X.shape) == 1:
        X = np.reshape(X, (1, -1))
    X_ = PolynomialFeatures(degree=max_order).fit_transform(X)
    X_ = np.concatenate([X_] + [f(X)
                                for f in extra_term_dicts.values()], axis=1)
    return X_



class ODE(object):
    def __init__(self, name='ode', dimension=3, max_order=2, variables=None, coef=None, h=0.01, extra_term_dicts=None):
        self.name = name
        self.dimension = dimension
        self.max_order = max_order
        if variables is None:
            if dimension == 1:
                self.variables = ['x']
            elif dimension == 2:
                self.variables = ['x', 'y']
            elif dimension == 3:
                self.variables = ['x', 'y', 'z']
            else:
                self.variables = [f'x{i+1}' for i in range(dimension)]
        else:
            if len(variables) != dimension:
                raise ValueError(f'Number of Variables ({len(variables)}) doesn\'t fit dimension ({dimension})!')
            else:
                self.variables = variables
            
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

        self.turb_mode = None
        self.turb_args = np.zeros(self.coef_shape + (2, ))
        self.h = h
        self._info = None

    def set_coef(self, eqn_index, term, value):
        if term in self.terms:
            self.coef_init[eqn_index, self.terms.index(term)] = value
            self.coef[eqn_index, self.terms.index(term)] = value
        elif term == '1' or '':
            self.coef_init[eqn_index, 0] = value
            self.coef[eqn_index, 0] = value
        else:
            raise KeyError(f'Term {term} not found!')
        
    def set_turb(self, turb_mode, turb_args):
        if turb_mode not in ['normal', 'uniform', 'rate']:
            raise ValueError('turb_mode should be \'normal\', \'uniform\' or \'rate\'!')
        if turb_args.shape != self.coef_shape + (2, ):
            raise ValueError(f'turb_args shape should be {self.coef_shape + (2, )}!')
        
        self.turb_mode = turb_mode
        self.turb_args = turb_args


    def d(self, points):
        points_ = get_basis(X=points, max_order=self.max_order, extra_term_dicts=self.extra_term_dicts)
        return points_ @ self.coef.T

    # def dd(self, points):
    #     dpoints = self.d(points)
        

    def R_K(self, point, h=None):
        if h is None:
            h = self.h
        point = np.array(point).reshape((1, -1))
        K1 = self.d(point)
        K2 = self.d(point + h / 2 * K1)
        K3 = self.d(point + h / 2 * K2)
        K4 = self.d(point + h * K3)
        return point + h / 6 * (K1 + 2*K2 + 2*K3 + K4)


    def random_coef(self):
        # if mode='normal', then turb_args means normal(mean, std) for every coef
        # if mode='uniform', then turb_args means uniform(low, high) for every coef
        if self.turb_mode == 'normal':
            self.coef = np.random.normal(
                loc=self.turb_args[:, :, 0], scale=self.turb_args[:, :, 1], size=self.coef_shape)
        elif self.turb_mode == 'uniform':
            self.coef = np.random.uniform(
                low=self.turb_args[:, :, 0], high=self.turb_args[:, :, 1], size=self.coef_shape)
        elif self.turb_mode == 'rate':
            self.coef = np.random.uniform(
                low=self.turb_args[:, :, 0] * (1-self.turb_args[:, :, 1]), high=self.turb_args[:, :, 0] * (1+self.turb_args[:, :, 1]), size=self.coef_shape)

    def reset_coef(self):
        self.coef = np.copy(self.coef_init)

    def exp(self, init_point=None, num_points=20000, omit_points=1000, h=None, bounds=None, return_status=False, epsilon=EPSILON):
        if h is None:
            h = self.h
        if init_point is None:
            init_point = np.random.normal(size=(1, self.dimension))
            for _ in range(10):
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

    def plot(self, points=None, init_point=None, num_points=20000, omit_points=1000, h=None, plot_axis=[0, 1, 2], subplot=111, exp_plot_save=False, pic_save_dir=None, plot_simple=False):
        if points is None:
            points = self.exp(init_point=init_point, num_points=num_points, omit_points=omit_points, h=h)

        plt.figure(figsize=(8, 8))

        if len(plot_axis) == 3:
            ax = plt.subplot(subplot, projection='3d')
            xs = points[:, plot_axis[0]]
            ys = points[:, plot_axis[1]]
            zs = points[:, plot_axis[2]]
            if plot_simple:
                ax.set_xticks([])  # 移除刻度
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_frame_on(False)  # 移除坐标轴框架
                ax.grid(False)  # 关闭网格
                ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 隐藏 X 轴
                ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 隐藏 Y 轴
                ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 隐藏 Z 轴
                ax.set_axis_off()  # 彻底移除所有轴和背景 
            plt.plot(xs, ys, zs, color = '#f79059')
            if exp_plot_save:
                plt.savefig(pic_save_dir + '.png', dpi=450, bbox_inches='tight', pad_inches=0)

        elif len(plot_axis) == 2:
            # plt.figure()
            plt.subplot(subplot)
            xs = points[:, plot_axis[0]]
            ys = points[:, plot_axis[1]]
            plt.plot(xs, ys)
            if exp_plot_save:
                plt.savefig(pic_save_dir + '.png', dpi=450)

        elif len(plot_axis) == 1:
            # plt.figure()
            plt.subplot(subplot)
            xs = points[:, plot_axis[0]]
            plt.plot(xs)
            if exp_plot_save:
                plt.savefig(pic_save_dir + '.png', dpi=450)

        else:
            raise ValueError('Plot Dimension should be 1, 2 or 3!')

    def show(self):
        plt.show()

    def savefig(self, file_name=None, save_dir='./'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if file_name is None:
            file_name = self.name + '.jpg'
        plt.savefig(save_dir + file_name)
        plt.close()

    def get_info(self, round_digits=3, epsilon=0.001):
        self._info = ''
        for i in range(self.dimension):
            # print(self.formula[i], '=', dict(zip(self.terms, self.coef[i])))
            self._info += f'd{self.variables[i]}/dt = ' 
            line = (' + '.join([(str(round(self.coef_init[i, j], round_digits)) + self.terms[j])
                                 for j in range(self.coef_shape[1]) if abs(self.coef_init[i, j]) > epsilon]))
            if line:
                self._info += line
            else:
                self._info += '0'
            self._info += '\n'
        return self._info

    @property
    def info(self):
        return self.get_info()


init_point = np.random.normal(size=(1, 3))
# init_point = np.array([-5.76,  2.27,  32.82])


# lorenz_tiny
lorenz_tiny = ODE(name='lorenz_tiny', dimension=3, max_order=2)
# dx
lorenz_tiny.set_coef(0, 'x', -10)
lorenz_tiny.set_coef(0, 'y', 10)
# dy
lorenz_tiny.set_coef(1, '', -13.5)
lorenz_tiny.set_coef(1, 'x', 28)
lorenz_tiny.set_coef(1, 'y', -1)
lorenz_tiny.set_coef(1, 'z', 25)
lorenz_tiny.set_coef(1, 'xz', -50)
# dz
lorenz_tiny.set_coef(2, '', 12.5)
lorenz_tiny.set_coef(2, 'x', -25)
lorenz_tiny.set_coef(2, 'y', -25)
lorenz_tiny.set_coef(2, 'z', -8/3)
lorenz_tiny.set_coef(2, 'xy', 50)
# turb
turb_mode = 'rate'
turb_args = np.zeros(lorenz_tiny.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(lorenz_tiny.coef_init)
turb_args[:, :, 1] = np.ones(lorenz_tiny.coef_shape) * 0.1
lorenz_tiny.set_turb(turb_mode, turb_args)


# chen_tiny
chen_tiny = ODE(name='chen_tiny')
# dx
chen_tiny.set_coef(0, 'x', -35)
chen_tiny.set_coef(0, 'y', 43.75)
chen_tiny.set_coef(0, '', -4.375)
# dy
chen_tiny.set_coef(1, 'x', -5.6)
chen_tiny.set_coef(1, 'y', 28)
chen_tiny.set_coef(1, 'z', 20)
chen_tiny.set_coef(1, 'xz', -40)
chen_tiny.set_coef(1, '', -11.2)
# dz
chen_tiny.set_coef(2, 'x', -80)
chen_tiny.set_coef(2, 'y', -80)
chen_tiny.set_coef(2, 'z', -3)
chen_tiny.set_coef(2, 'xy', 160)
chen_tiny.set_coef(2, '', 40)
# turb
turb_mode = 'rate'
turb_args = np.zeros(chen_tiny.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(chen_tiny.coef_init)
turb_args[:, :, 1] = np.ones(chen_tiny.coef_shape) * 0.1
chen_tiny.set_turb(turb_mode, turb_args)


# lv_tiny
lv_tiny = ODE(name='lv_tiny')
# dx
lv_tiny.set_coef(0, 'x', -36)
lv_tiny.set_coef(0, 'y', 36)
# dy
lv_tiny.set_coef(1, 'x', 1)
lv_tiny.set_coef(1, 'y', 20)
lv_tiny.set_coef(1, 'z', 25)
lv_tiny.set_coef(1, '', -10.803)
lv_tiny.set_coef(1, 'xz', -50)
# dz
lv_tiny.set_coef(2, 'x', -25)
lv_tiny.set_coef(2, 'y', -25)
lv_tiny.set_coef(2, 'z', -3)
lv_tiny.set_coef(2, '', 12.5)
lv_tiny.set_coef(2, 'xy', 50)
# turb
turb_mode = 'rate'
turb_args = np.zeros(lv_tiny.coef_shape + (2, ))
turb_args[:, :, 0] = np.copy(lv_tiny.coef_init)
turb_args[:, :, 1] = np.ones(lv_tiny.coef_shape) * 0.1
lv_tiny.set_turb(turb_mode, turb_args)


'''
Def ode(seirs)
'''
seirs = ODE(name='seirs', variables=['S', 'E', 'I'], max_order=2,
          extra_term_dicts={'R': R})


seirs._beta = 0.9  # 感染率，即一个感染者在单位时间内感染易感者的概率
seirs._sigma = 0.1  # 潜伏期倒数，即一个暴露者转变为感染者的速率
seirs._gamma = 0.2  # 康复率，即一个感染者在单位时间内康复并获得免疫力的概率
seirs._xi = 0.15  # 失去免疫率，即一个康复者在单位时间内失去免疫力的概率
seirs._N = 1  # 初始总人数

seirs.set_coef(0, 'SI', -seirs._beta/seirs._N)
seirs.set_coef(0, 'R', seirs._xi)

seirs.set_coef(1, 'E', -seirs._sigma)
seirs.set_coef(1, 'SI', seirs._beta/seirs._N)

seirs.set_coef(2, 'I', -seirs._gamma) 
seirs.set_coef(2, 'E', seirs._sigma)

turb_mode = 'uniform'
turb_args = np.zeros(seirs.coef_shape + (2, ))
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
    return R / (R+CR0+EPSILON)


def C_(X):
    R, C, P = X[:, [0]], X[:, [1]], X[:, [2]]
    return C / (C+CC0+EPSILON)


def CR_(X):
    R, C, P = X[:, [0]], X[:, [1]], X[:, [2]]
    return C * R_(X)


def PC_(X):
    R, C, P = X[:, [0]], X[:, [1]], X[:, [2]]
    return P * C_(X)


rcp_extra_term_dicts = {'CR_': CR_, 'PC_': PC_}
rcp = ODE(name='rcp', variables=['R', 'C', 'P'],
          extra_term_dicts=rcp_extra_term_dicts)

rcp.set_coef(0, 'R', 1)
rcp.set_coef(0, 'RR', -1/CK)
rcp.set_coef(0, 'CR_', -CXC * CYC)

rcp.set_coef(1, 'CR_', CXC * CYC)
rcp.set_coef(1, 'C', -CXC)
rcp.set_coef(1, 'PC_', -CXP * CYP)

rcp.set_coef(2, 'PC_', CXP * CYP)
rcp.set_coef(2, 'P', -CXP)

turb_mode = 'rate'
turb_args = np.zeros(rcp.coef_shape + (2, ))
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