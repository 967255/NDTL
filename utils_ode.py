import os
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


INFINITY = 1e5
EPSILON = 1e-6


'''
For ode
'''
# 提取ode阶数
# tmp: get_orders(3) --> [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
def get_orders(n_variables, order=2):
    '''tmp: get_orders(3) --> [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]'''
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


# return [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
def form_orders(n_variables=3, max_order=2):
    '''return [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]'''
    orders = []
    for order in range(max_order+1):
        orders += get_orders(n_variables, order)
    return orders


# ['', 'x', 'y', 'z', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']
def get_terms(variables='xyz', max_order=2, extra_term_dicts={}):
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


# (core)???疑似sindy拟合出现的多项式基
def get_basis(X, max_order=2, extra_term_dicts={}):
    X = np.array(X)
    if len(X.shape) == 1:
        X = np.reshape(X, (1, -1))
    X_ = PolynomialFeatures(degree=max_order).fit_transform(X)
    X_ = np.concatenate([X_] + [f(X)
                                for f in extra_term_dicts.values()], axis=1)
    return X_


# return vec
# input: ode is a class object, grids is a list;
# input_tmp: ode=lorenz_shifted, grids=[(48,), (48,), (48,)]
# output_tmp: vec: type=numpy.ndarray, shape=(48, 48, 48, 3)
def ode2vec(ode, grids):
    X = calc_X(grids)
    vec = ode.d(X).reshape([len(grid) for grid in grids] + [len(grids)])
    return vec


# 4阶runge-kutta方法的实现 point:初始点 vec:向量场 h:步长 (可能需要连续向量场)
def R_K_vec(point, vec, pad_width=0, h=0.0001):
    point = np.array(point).reshape((1, -1))
    vec_padded = np.pad(vec, ((pad_width, pad_width),) * 3 + ((0, 0), ), mode='constant', constant_values=0)

    x = np.linspace(0, 1, vec.shape[0])
    y = np.linspace(0, 1, vec.shape[1])
    z = np.linspace(0, 1, vec.shape[2])

    if pad_width > 0:
        x_padded = np.linspace(x[0] - (x[1] - x[0]), x[-1] + (x[1] - x[0]), vec.shape[0] + 2 * pad_width)
        y_padded = np.linspace(y[0] - (y[1] - y[0]), y[-1] + (y[1] - y[0]), vec.shape[1] + 2 * pad_width)
        z_padded = np.linspace(z[0] - (z[1] - z[0]), z[-1] + (z[1] - z[0]), vec.shape[2] + 2 * pad_width)
        vector_field_interpolator = RegularGridInterpolator((x_padded, y_padded, z_padded), vec_padded, bounds_error=True, method='linear')
    else:
        vector_field_interpolator = RegularGridInterpolator((x, y, z), vec, bounds_error=True, method='linear')

    K1 = vector_field_interpolator(point)
    K2 = vector_field_interpolator(point + h / 2 * K1)
    K3 = vector_field_interpolator(point + h / 2 * K2)
    K4 = vector_field_interpolator(point + h * K3)
    return point + h / 6 * (K1 + 2*K2 + 2*K3 + K4)


# return ODE(variables=variables, name=name, coef=coef, max_order=max_order, dimension=dimension, extra_term_dicts=extra_term_dicts)
def vec2ode(vec, grids, sparse_coef=None, max_order=2, extra_term_dicts={}, mode='linear', alpha=0, fit_intercept=True, name='ode', variables=None):
    dimension = len(grids)
    Y = vec.reshape((-1, dimension))
    X = calc_X(grids)

    assert X.shape[0] == Y.shape[0], f'X.shape[0] ({X.shape[0]}) != Y.shape[0] ({Y.shape[0]})'

    if sparse_coef is None:
        positive = False
    else:
        positive = True

    X_ = get_basis(X=X, max_order=max_order, extra_term_dicts=extra_term_dicts)
    if mode == 'linear':
        clf = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    elif mode == 'lasso':
        clf = Lasso(alpha=alpha, fit_intercept=fit_intercept, positive=positive)
    elif mode == 'ridge':
        clf = Ridge(alpha=alpha, fit_intercept=fit_intercept, positive=positive)
    else:
        raise ValueError('mode should be linear, lasso or ridge')
    
    if sparse_coef is None:
        coef = np.zeros((dimension, X_.shape[1]))
        clf.fit(X_[:, 1:], Y)
        coef[:, 1:] = clf.coef_
        if fit_intercept:
            coef[:, 0] = clf.intercept_
        
    else:
        coef = np.zeros((dimension, X_.shape[1]))
        assert coef.shape == sparse_coef.shape
        for i in range(coef.shape[0]):
            X_sp = X_ * sparse_coef[i, :]
            clf.fit(X_sp, Y[:, i])
            coef[i, :] = clf.coef_
            if fit_intercept:
                coef[i, 0] = clf.intercept_
    
        coef = coef * sparse_coef

    return ODE(variables=variables, name=name, coef=coef, max_order=max_order, dimension=dimension, extra_term_dicts=extra_term_dicts)


# point.shape = (len, dim), maybe(len=1000, dim=3)
# return limits = [[第一列min, 第一列max], [第二列min, 第二列max], ... ,[最后一列min, 最后一列max]]
def get_limits(points):
    limits = [[points[:, i].min(), points[:, i].max()] for i in range(points.shape[1])]
    return limits


# get grids
# return grids
# input_tmp: grid_limits_target = ((0, 1), (0, 1), (0, 1))
# input_tmp: grid_shape = (48, 48, 48)
# output_tmp: type=list, len=3, output[0]=numpy.ndarray, output[0].shape=(48,)
def calc_grids(grid_limits, grid_shape):
    grids = [np.linspace(start=grid_limits[i][0], stop=grid_limits[i][1],
                    num=grid_shape[i]) for i in range(len(grid_shape))]
    return grids


# input_tmp: grids:a list, [(48,), (48,), (48)] return X.shape = (-1, len(grid)) 
# rmk: you can get grids from calc_grids()
# output_tmp: X: type=numpy.ndarray, shape=(110592=48^3, 3)
def calc_X(grids):
    X = np.array([x for x in product(*grids)]).reshape((-1, len(grids)))
    return X


# ode_origin --> ode_target(a real ode, not vector field)
def ode_change_grid(ode_origin, grid_limits_origin, grid_limits_target=None, grid_shape=None, max_order=2):
    # grid_limits_origin = ((-25, 25), (-25, 25), (0, 50))
    if grid_limits_target is None:
        grid_limits_target = ((0, 1), (0, 1), (0, 1))

    if grid_shape is None:
        grid_shape = (48, 48, 48)

    grids_origin = calc_grids(grid_limits_origin, grid_shape)
    grids_target = calc_grids(grid_limits_target, grid_shape)

    vec = ode2vec(ode=ode_origin, grids=grids_origin)

    ode_target = vec2ode(vec, grids=grids_target, max_order=max_order, extra_term_dicts={}, name=ode_origin.name)
    
    if ode_origin.turb_mode == 'rate':
        turb_mode = 'rate'
        turb_args = np.zeros_like(ode_origin.turb_args)
        turb_args[:, :, 0] = np.copy(ode_target.coef_init)
        turb_args[:, :, 1] = np.copy(ode_origin.turb_args[:, :, 1])
        ode_target.set_turb(turb_mode, turb_args)

    return ode_target


def parse_ode_equations(equation_str):
    """
    将类似于：
    'dx/dt = 11.889 + -11.77x + 12.366y + -1.031z + 0.016xx + ...\n
     dy/dt = ...\n
     dz/dt = ...\n'
    的字符串解析为可用在 solve_ivp 中的 Python 函数 f(t, w)。
    返回值: 一个函数 f(t, w)，其中 w=[x, y, z]。
    """
    # 1) 按行拆分
    lines = equation_str.strip().split('\n')
    if len(lines) < 3:
        raise ValueError("至少需要3行 (dx, dy, dz)，请检查输入字符串")
    
    dx_str = lines[0].split('=')[1].strip()
    dy_str = lines[1].split('=')[1].strip()
    dz_str = lines[2].split('=')[1].strip()
    
    # 2) 插入显式乘法：当数字（包括符号、小数、科学计数法）后紧跟 x, y, z 或左括号时插入 *
    pattern = re.compile(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=[xyz\(])')
    dx_str = pattern.sub(r'\1*', dx_str)
    dy_str = pattern.sub(r'\1*', dy_str)
    dz_str = pattern.sub(r'\1*', dz_str)
    
    # 3) 替换复合变量：先把 xx, xy, xz, yy, yz, zz 替换为对应的 w 表达式
    replacements = {
        'xx': 'w[0]*w[0]',
        'xy': 'w[0]*w[1]',
        'xz': 'w[0]*w[2]',
        'yy': 'w[1]*w[1]',
        'yz': 'w[1]*w[2]',
        'zz': 'w[2]*w[2]'
    }
    for key, val in replacements.items():
        dx_str = dx_str.replace(key, val)
        dy_str = dy_str.replace(key, val)
        dz_str = dz_str.replace(key, val)
    
    # 4) 替换剩余的单变量 x, y, z（利用正则确保只替换独立变量）
    dx_str = re.sub(r'\bx\b', 'w[0]', dx_str)
    dx_str = re.sub(r'\by\b', 'w[1]', dx_str)
    dx_str = re.sub(r'\bz\b', 'w[2]', dx_str)
    
    dy_str = re.sub(r'\bx\b', 'w[0]', dy_str)
    dy_str = re.sub(r'\by\b', 'w[1]', dy_str)
    dy_str = re.sub(r'\bz\b', 'w[2]', dy_str)
    
    dz_str = re.sub(r'\bx\b', 'w[0]', dz_str)
    dz_str = re.sub(r'\by\b', 'w[1]', dz_str)
    dz_str = re.sub(r'\bz\b', 'w[2]', dz_str)
    
    # 调试时可以打印结果，检查生成的表达式是否符合预期
    # print("dx_str =", dx_str)
    # print("dy_str =", dy_str)
    # print("dz_str =", dz_str)
    
    # 5) 编译成 Python 表达式
    dx_compiled = compile(dx_str, '<string>', 'eval')
    dy_compiled = compile(dy_str, '<string>', 'eval')
    dz_compiled = compile(dz_str, '<string>', 'eval')
    
    # 6) 返回 f(t, w)
    def f(t, w):
        return [eval(dx_compiled, {"w": w, "np": np}),
                eval(dy_compiled, {"w": w, "np": np}),
                eval(dz_compiled, {"w": w, "np": np})]
    
    return f

'''
def plot_3d_ode(f, w0, t_span=(0, 100), num_points=10000, method='RK45', plot_simple=True, plot_save=False, pic_save_dir=None, **solve_ivp_kwargs):
    """
    对三维 ODE 进行积分并绘制轨迹图像。
    
    参数：
    --------
    f : function
        ODE 的右端函数 f(t, w)，其中 w = [x, y, z]
    w0 : array_like
        初始条件，例如 [x0, y0, z0]
    t_span : tuple
        积分时间区间 (t0, tf)，默认为 (0, 100)
    num_points : int
        用于 t_eval 的积分点数量，默认 10000
    method : str
        传递给 solve_ivp 的积分方法(例如 'RK45')
    solve_ivp_kwargs : dict
        其他传递给 solve_ivp 的参数，例如 max_step、rtol、atol 等

    返回：
    --------
    None, 函数直接显示绘图
    """
    # 生成积分时间点
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # 数值积分
    sol = solve_ivp(f, t_span, w0, method=method, t_eval=t_eval, **solve_ivp_kwargs)
    
    if not sol.success:
        print("积分失败：", sol.message)
        return
    
    # 提取解向量
    x, y, z = sol.y[0, :], sol.y[1, :], sol.y[2, :]
    
    # 绘图
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=0.5, color = 'black')
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
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.set_title("3D Trajectory of the ODE")
    if plot_save:
        # plt.savefig(pic_save_dir + '.png', dpi=450, bbox_inches='tight', pad_inches=0)
        plt.savefig(pic_save_dir + '.svg', bbox_inches='tight', pad_inches=0)
        
    plt.show()
'''

def plot_3d_ode(f, w0, t_span=(0, 100), num_points=10000, method='RK45', plot_simple=True, plot_save=False, pic_save_dir=None, **solve_ivp_kwargs):
    """
    Integrate a 3D ODE and plot the trajectory.
    
    Parameters:
    --------
    f : function
        The right-hand side of the ODE, f(t, w), where w = [x, y, z]
    w0 : array_like
        The initial state, e.g. [x0, y0, z0]
    t_span : tuple
        Time span for integration (t0, tf), default is (0, 100)
    num_points : int
        Number of integration points for t_eval, default is 10000
    method : str
        Integration method to pass to solve_ivp (e.g. 'RK45')
    solve_ivp_kwargs : dict
        Additional arguments for solve_ivp (e.g., max_step, rtol, atol)
        
    Returns:
    --------
    None. The function displays the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Generate evaluation points
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # Numerical integration
    sol = solve_ivp(f, t_span, w0, method=method, t_eval=t_eval, **solve_ivp_kwargs)
    
    if not sol.success:
        print("Integration failed:", sol.message)
        return
    
    # Extract solution vectors
    x, y, z = sol.y[0, :], sol.y[1, :], sol.y[2, :]
    
    # Plotting
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=0.5, color='#BF1E2E')
    
    if plot_simple:
        # Remove ticks and tick labels, as well as axis labels,
        # but keep the background and grid.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.tick_params(axis='x', pad=0)
        # ax.tick_params(axis='y', pad=0)
        # ax.tick_params(axis='z', pad=0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
    if plot_save:
        # Save the figure with a transparent background.
        # Uncomment the desired format.
        # plt.savefig(pic_save_dir + '.png', dpi=450, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.savefig(pic_save_dir + '.svg', bbox_inches='tight', pad_inches=0, transparent=True)
        
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_power_spectrum(f, w0, t_span=(0, 100), num_points=10000, variable_idx=0, freq_range=(0,30), method='RK45',plot_simple=True, plot_save=False, pic_save_dir=None, **solve_ivp_kwargs):
    """
    计算并绘制三维 ODE 系统中指定变量在给定频率范围内的功率谱。
    
    参数：
    --------
    f : function
        ODE 右端函数 f(t, w)，其中 w = [x, y, z]
    w0 : array_like
        初始条件，例如 [x0, y0, z0]
    t_span : tuple
        积分时间区间 (t0, tf)，默认 (0, 100)
    num_points : int
        用于 t_eval 的点数，决定时间分辨率，默认 10000
    variable_idx : int 或 list 或 None
        指定要绘制功率谱的变量下标。如果为 int, 则绘制该变量的功率谱;
        如果为 list, 则分别绘制列表中各个变量的功率谱;
        如果为 None, 则绘制所有变量。
    freq_range : tuple
        绘制的频率范围，默认为 (0, 30)
    method : str
        传递给 solve_ivp 的积分方法(例如 'RK45')
    solve_ivp_kwargs : dict
        其他传递给 solve_ivp 的参数，例如 max_step、rtol、atol 等
        
    返回：
    --------
    None。函数直接绘制图像。
    """
    # 生成积分时间点
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # 数值积分
    sol = solve_ivp(f, t_span, w0, method=method, t_eval=t_eval, **solve_ivp_kwargs)
    
    if not sol.success:
        print("积分失败：", sol.message)
        return
    
    # 提取解向量，sol.y 的形状为 (n, num_points)，n 为系统维数
    time_series = sol.y
    dt = (t_span[1] - t_span[0]) / (num_points - 1)
    
    # 如果 variable_idx 是 int，则转换为列表；如果为 None，则绘制所有变量
    if variable_idx is None:
        variable_idx = list(range(len(w0)))
    elif isinstance(variable_idx, int):
        variable_idx = [variable_idx]
    
    # 绘图
    num_plots = len(variable_idx)
    fig, axs = plt.subplots(num_plots, 1, figsize=(4, 3*num_plots))
    if num_plots == 1:
        axs = [axs]
    
    for i, idx in enumerate(variable_idx):
        series = time_series[idx]
        # 去除均值，避免直流分量主导
        series = series - np.mean(series)
        # 计算 FFT
        fft_vals = np.fft.fft(series)
        power = np.abs(fft_vals)**2
        freqs = np.fft.fftfreq(len(series), d=dt)
        # 只取正频率且在 freq_range 内
        pos_mask = (freqs > freq_range[0]) & (freqs < freq_range[1])
        freqs = freqs[pos_mask]
        power = power[pos_mask]
        
        axs[i].plot(freqs, power, lw=0.8, color='black')
        axs[i].tick_params(direction='in')
        if plot_simple:
            # 下面这些方法适用于 2D Axes
            axs[i].set_ylim(0, 8e8)
            axs[i].set_xticks([])       # 移除 X 刻度
            axs[i].set_yticks([])       # 移除 Y 刻度
            # axs[i].grid(False)          # 关闭网格
        else:
            # 设置 X 轴在 freq_range 内稀疏的刻度，例如 6 个刻度
            axs[i].set_xticks(np.linspace(freq_range[0], freq_range[1], 3))
            # 设置 Y 轴刻度为当前功率范围内稀疏的刻度（这里用 np.min(power) 和 np.max(power)）
            axs[i].set_ylim(0, 8e8)
            axs[i].set_yticks(np.linspace(0, 8e8, 3))
            # 调大刻度标签的字体大小，例如设置为 14
            axs[i].tick_params(axis='both', which='major', labelsize=20, colors='black')
            axs[i].yaxis.get_offset_text().set_fontsize(20)
            # axs[i].set_title(f"Power Spectrum of Variable w[{idx}] (Freq {freq_range[0]}-{freq_range[1]})")
        axs[i].grid(True)
    
    plt.tight_layout()
    if plot_save:
        plt.savefig(pic_save_dir + '.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


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


'''
For SERIS
'''
def R(X, N=1):
    S, E, I = X[:, [0]], X[:, [1]], X[:, [2]]
    return N - S - E - I
