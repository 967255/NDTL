from utils_ode import *
from ode import *


'''
calculate physical quantity
'''
import numpy as np
import matplotlib.pyplot as plt

from math import log
from scipy.integrate import solve_ivp


'''
if __name__ == "__main__":
    # 第一步：解析字符串 ODE，得到 f(t, w)
    f = parse_ode_equations(equation_str)
    
    # 第二步：使用 Benettin 方法计算最大 Lyapunov 指数
    w1_0 = [1.0, 1.0, 1.0]  # 任选一个初始条件
    lyap_est = max_lyapunov_benettin(f, w1_0, delta0=1e-8, T=0.5, steps=500)
    
    print("Estimated Maximum Lyapunov Exponent:", lyap_est)

'''
def max_lyapunov_benettin(f, 
                          w1_0, 
                          delta0=1e-8, 
                          T=0.5, 
                          steps=500, 
                          method='RK45'):
    """
    用 Benettin 方法估计最大 Lyapunov 指数。
    参数：
    ----------
    f : function
        ODE 右端函数 f(t, w)，其中 w=[x, y, z,...]，返回同维度向量
    w1_0 : array_like
        主轨道的初始条件（如 [x0, y0, z0]）
    delta0 : float
        初始扰动向量的范数
    T : float
        每次积分的时间长度
    steps : int
        重复正则化的次数
    method : str
        传给 solve_ivp 的求解方法，如 'RK45', 'RK23', 'BDF' 等
    返回值：
    ----------
    lyap : float
        估计得到的最大 Lyapunov 指数
    """

    import numpy as np

    w1 = np.array(w1_0, dtype=float)
    
    # 给扰动一个随机方向，但范数固定为 delta0
    rand_dir = np.random.normal(size=len(w1_0))
    rand_dir /= np.linalg.norm(rand_dir)
    w2 = w1 + rand_dir * delta0
    
    sum_log_dist = 0.0
    
    for _ in range(steps):
        # 在区间 [0, T] 上分别对 w1, w2 积分
        sol1 = solve_ivp(f, [0, T], w1, method=method,
                         max_step=1e-2, rtol=1e-8, atol=1e-10)
        sol2 = solve_ivp(f, [0, T], w2, method=method,
                         max_step=1e-2, rtol=1e-8, atol=1e-10)
        
        w1_end = sol1.y[:, -1]
        w2_end = sol2.y[:, -1]
        
        # 计算演化结束后，两条轨道之间的距离
        dist = np.linalg.norm(w2_end - w1_end)
        
        # 累加 ln(dist / delta0)
        sum_log_dist += log(dist / delta0)
        
        # 重新把 w2_end 拉回到 w1_end 附近，保持距离为 delta0
        direction = w2_end - w1_end
        norm_dir = np.linalg.norm(direction)
        
        # 若 dist 很小，需保护一下，防止数值出错
        if norm_dir < 1e-16:
            direction = np.random.normal(size=len(w1_0))
            norm_dir = np.linalg.norm(direction)
        
        direction /= norm_dir  # 单位向量
        w1 = w1_end
        w2 = w1_end + direction * delta0
    
    # 每次迭代积分时间为 T，总共 steps 次；所以总演化时间是 steps * T
    lyap = sum_log_dist / (steps * T)
    return lyap


def max_lyapunov_exponent(init_point, ode, num_points=10000, omit_points=0, delta0=1e-6, dt=0.01, dim=3):
    """
    Compute the maximum Lyapunov exponent of the ODE system at a given initial point.
    
    Parameters:
    - init_point: Initial point
    - num_point: Number of time steps for ODE solution
    - ode: Function representing the ODE system
    - exp: Function used to return the ODE solution
    - delta0: Magnitude of the initial perturbation
    - dt: Time step size

    Returns:
    - Estimated value of the maximum Lyapunov exponent
    """
    
    # Use the exp function to obtain the solution of the ODE
    solution = ode.exp(init_point=init_point, num_points=num_points, omit_points=omit_points, h=0.01, bounds=None, return_status=False)
    
    # 计算系统的维度
    n = dim
    
    # 初始化扰动矩阵，每列代表一个方向的扰动
    perturbation_matrix = np.eye(n) * delta0  # delta0是扰动幅度
    
    # 计算Lyapunov指数
    lyapunov_exp = []
    
    # 遍历每个方向的扰动
    for j in range(n):
        # 对每个方向的初始扰动进行扰动计算
        perturbed_point = init_point + perturbation_matrix[j]
        
        # 获取扰动初始点的ODE解
        perturbed_solution = ode.exp(init_point=perturbed_point, num_points=num_points, omit_points=omit_points, h=0.01, bounds=None, return_status=False)
        
        # 计算扰动随时间的变化
        delta = np.linalg.norm(solution - perturbed_solution, axis=1)
        
        # 计算对数变化率并存储
        for i in range(1, num_points):
            lyapunov_exp.append(np.log(delta[i] / delta0) / (i * dt))
    
    # 返回最大Lyapunov指数估算值
    return np.mean(lyapunov_exp)



# 假设 exp() 已经可以计算ODE的解
# init_points: 初始点，num_points: 时间步数，ode: ODE函数
def compute_power_spectrum(ode, init_points, target_dim=0, num_points=10000, omit_points=0, dt=0.01, plot_show=False, plot_save_dir=None):
    # 步长为 0.01，使用RK4方法进行数值求解
    solution = ode.exp(init_point=init_points, num_points=num_points, omit_points=omit_points, h=dt, bounds=None, return_status=False)
    
    # 假设解是一个 N x M 形状的数组，其中 N 是时间步数，M 是解的维度
    # 提取你关心的某个分量（例如解的第一个分量）
    if target_dim not in [0, 1, 2]:
        print('targei_dim wrong!')
        
        exit()
    
    data = solution[:, target_dim]
    
    # 转换为 numpy 数组
    data = np.array(data)
    
    # 执行傅里叶变换
    freq = np.fft.fftfreq(len(data), dt)
    fft_values = np.fft.fft(data)
    
    # 计算功率谱（取模的平方）
    power_spectrum = np.abs(fft_values)**2
    
    # 只保留正频率部分
    positive_freqs = freq[freq >= 0]
    positive_power_spectrum = power_spectrum[freq >= 0]
    
    if plot_show:
    # 绘制功率谱
        plt.figure(figsize=(8, 2))
        plt.plot(positive_freqs, positive_power_spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum')
        plt.title('Power Spectrum of ODE Solution')
        plt.show()

    if plot_save_dir != None:
        plt.figure(figsize=(8, 2))
        plt.plot(positive_freqs, positive_power_spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum')
        plt.title('Power Spectrum of ODE Solution')
        plt.savefig(plot_save_dir + '.png')

    return positive_freqs, positive_power_spectrum



'''
For verify
'''
def r_simplify(ode, grids, fit_intercept=True):
    vec = ode2vec(ode, grids)
    ode_simplified = vec2ode(vec, grids, variables=['S', 'E', 'I'], name=f'{ode.name}_simplified', max_order=ode.max_order, extra_term_dicts={}, fit_intercept=fit_intercept)
    # ode_simplified = vec2ode(vec, grids, variables=seirs.variables, name=f'{ode.name}_simplified', max_order=ode.max_order, extra_term_dicts={}, fit_intercept=fit_intercept)
    return ode_simplified

# ode_shifted_list.append(seirs)



def calc_dist(coef1, coef2, normalization=True):
    assert coef1.shape == coef2.shape
    if normalization:
        return np.sqrt(np.mean(np.square((coef1-coef2)/(coef2))))  
    else:
        return np.sqrt(np.mean(np.square(coef1-coef2)))


def calc_corr(coef1, coef2):
    assert coef1.shape == coef2.shape
    corr = np.corrcoef(coef1.flatten(), coef2.flatten())[0, 1]
    return corr
    
    
    

def exp_vec(vec, init_point=None, num_points=20000, omit_points=1000, pad_width=0, h=0.0001, return_status=False, epsilon=EPSILON):
    n_dimension = vec.shape[-1]
    if init_point is None:
        init_point = np.random.normal(size=(1, n_dimension))
        for _ in range(10):
            init_point = R_K_vec(init_point, vec, pad_width=pad_width, h=h)
    else:
        init_point = np.array(init_point).reshape((1, -1))
    points = [init_point]
    for i in range(1, num_points):
        try:
            points.append(R_K_vec(points[-1], vec, pad_width=pad_width, h=h))
        except ValueError:
            print(f'Point {i} Out of Bounds!')
            status = f'Out of Bounds!'
            break
        if np.sum(np.abs(points[-1]-points[-2])) < epsilon:
            print(f'Value Converged at Point {i} at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})')
            status = f'Converged at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})'
            break
        if np.abs(points[-1]).max() > INFINITY:
            print(f'Value Exploded at Point {i}!')
            status = f'Exploded!'
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



def find_nearest_point(grids, point):
    x_grid, y_grid, z_grid = grids
    idx_x = np.abs(x_grid - point[0]).argmin()
    idx_y = np.abs(y_grid - point[1]).argmin()
    idx_z = np.abs(z_grid - point[2]).argmin()
    return (idx_x, idx_y, idx_z)


def R_K_direct(vec, grids, point, h=0.0001):
    k1 = h * vec[find_nearest_point(grids, point)]
    k2 = h * vec[find_nearest_point(grids, point + 0.5 * k1)]
    k3 = h * vec[find_nearest_point(grids, point + 0.5 * k2)]
    k4 = h * vec[find_nearest_point(grids, point + k3)]
    return np.array(point + (k1 + 2 * k2 + 2 * k3 + k4) / 6).reshape((1, -1))


def exp_vec_direct(vec, grids, init_point=None, num_points=20000, omit_points=1000, h=0.0001, return_status=False, use_bounds=True, local_tor=500, epsilon=EPSILON):
    n_dimension = vec.shape[-1]
    if init_point is None:
        init_point = np.random.normal(size=(1, n_dimension))
        for _ in range(10):
            init_point = R_K_direct(vec, grids, init_point[0], h=h)
    else:
        init_point = np.array(init_point).reshape((1, -1))
    points = [init_point]

    min_interval = 2*np.min(np.diff(grids))
    bounds = np.array([[grids[0].min()-min_interval, grids[0].max()+min_interval], [grids[1].min()-min_interval, grids[1].max()+min_interval], [grids[2].min()-min_interval, grids[2].max()+min_interval]])
    for i in range(1, num_points):
        try:
            points.append(R_K_direct(vec, grids, points[-1][0], h=h))
        except ValueError:
            print(f'Point {i} Out of Bounds!')
            status = f'Out of Bounds!'
            break
        if points[-1][0, 0] < bounds[0, 0] or points[-1][0, 0] > bounds[0, 1] or \
            points[-1][0, 1] < bounds[1, 0] or points[-1][0, 1] > bounds[1, 1] or \
            points[-1][0, 2] < bounds[2, 0] or points[-1][0, 2] > bounds[2, 1]:
            if use_bounds:
                print(f'Point {i} Out of Bounds!')
                status = f'Out of Bounds!'
                break
            else:
                print(f'Warning: Point {i} Out of Bounds!')
        if np.sum(np.abs(points[-1]-points[-2])) < epsilon:
            print(f'Value Converged at Point {i} at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})')
            status = f'Converged at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})'
            break
        if np.abs(points[-1]).max() > INFINITY:
            print(f'Value Exploded at Point {i}!')
            status = f'Exploded!'
            break
        if i > local_tor:
            grid_points = set([find_nearest_point(grids, points[-i][0]) for i in range(1, local_tor)])
            if len(grid_points) < 5:
                points = points[:-local_tor+1]
                print(f'Value Converged at Point {len(points)} at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})')
                status = f'Converged at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})'
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




def dSdt_neg(point, ode, clip=False):
    dS_neg = ode.coef[0, :] < 0
    points = get_basis(X=np.array(point).reshape((1, -1)), max_order=2, extra_term_dicts=seirs.extra_term_dicts)
    if clip:
        return (np.clip(points, 0, 1) @ (ode.coef[0, :] * dS_neg))[0]
    else:
        return (points @ (ode.coef[0, :] * dS_neg))[0]

def dIdt_neg(point, ode, clip=False):
    dI_neg = ode.coef[2, :] < 0
    points = get_basis(X=np.array(point).reshape((1, -1)), max_order=2, extra_term_dicts=seirs.extra_term_dicts)
    if clip:
        return (np.clip(points, 0, 1) @ (ode.coef[2, :] * dI_neg))[0]
    else:
        return (points @ (ode.coef[2, :] * dI_neg))[0]

def calc_r0(ode, init_point=None, h=0.0001, num_points=100, clip=False):
    if init_point is None:
        init_point = np.array((0.99, 0, 0.01)).reshape((1, -1)) 
    
    points = ode.exp(init_point, h=h, num_points=num_points, omit_points=0, return_status=False, epsilon=1e-8)

    # gamma = -ode.coef[2, 3]  # dI/dt 中的 I 项系数 -gamma

    S_decrease = 0
    I_decrease = 0
    for i in range(num_points):
        point = points[i]
        S_decrease += dSdt_neg(point, ode, clip=clip) * h
        I_decrease += dIdt_neg(point, ode, clip=clip) * h
    
    beta_ = -S_decrease / (init_point[0, 2] * num_points * h)
    gamma_ = -I_decrease / (init_point[0, 2] * num_points * h)
    r0 = beta_ / gamma_
    print(f'beta_ = {beta_: .3f}, gamma_ = {gamma_: .3f}, r0 = {r0: .3f}')

    return r0

