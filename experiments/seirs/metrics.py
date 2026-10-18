from experiments.seirs.fields import EPSILON, INFINITY, R_K_vec, ode2vec, vec2ode

"""
calculate physical quantity
"""
import numpy as np
import matplotlib.pyplot as plt

from math import log
from scipy.integrate import solve_ivp


def max_lyapunov_benettin(f, w1_0, delta0=1e-8, T=0.5, steps=500, method="RK45"):
    "Estimate the largest Lyapunov exponent by Benettin renormalization of paired trajectories."

    import numpy as np

    w1 = np.array(w1_0, dtype=float)

    rand_dir = np.random.normal(size=len(w1_0))
    rand_dir /= np.linalg.norm(rand_dir)
    w2 = w1 + rand_dir * delta0

    sum_log_dist = 0.0

    for _ in range(steps):

        sol1 = solve_ivp(
            f, [0, T], w1, method=method, max_step=1e-2, rtol=1e-8, atol=1e-10
        )
        sol2 = solve_ivp(
            f, [0, T], w2, method=method, max_step=1e-2, rtol=1e-8, atol=1e-10
        )

        w1_end = sol1.y[:, -1]
        w2_end = sol2.y[:, -1]

        dist = np.linalg.norm(w2_end - w1_end)

        sum_log_dist += log(dist / delta0)

        direction = w2_end - w1_end
        norm_dir = np.linalg.norm(direction)

        if norm_dir < 1e-16:
            direction = np.random.normal(size=len(w1_0))
            norm_dir = np.linalg.norm(direction)

        direction /= norm_dir
        w1 = w1_end
        w2 = w1_end + direction * delta0

    lyap = sum_log_dist / (steps * T)
    return lyap


def max_lyapunov_exponent(
    init_point, ode, num_points=10000, omit_points=0, delta0=1e-6, dt=0.01, dim=3
):
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
    solution = ode.exp(
        init_point=init_point,
        num_points=num_points,
        omit_points=omit_points,
        h=0.01,
        bounds=None,
        return_status=False,
    )

    n = dim

    perturbation_matrix = np.eye(n) * delta0

    lyapunov_exp = []

    for j in range(n):

        perturbed_point = init_point + perturbation_matrix[j]

        perturbed_solution = ode.exp(
            init_point=perturbed_point,
            num_points=num_points,
            omit_points=omit_points,
            h=0.01,
            bounds=None,
            return_status=False,
        )

        delta = np.linalg.norm(solution - perturbed_solution, axis=1)

        for i in range(1, num_points):
            lyapunov_exp.append(np.log(delta[i] / delta0) / (i * dt))

    return np.mean(lyapunov_exp)


def compute_power_spectrum(
    ode,
    init_points,
    target_dim=0,
    num_points=10000,
    omit_points=0,
    dt=0.01,
    plot_show=False,
    plot_save_dir=None,
):

    solution = ode.exp(
        init_point=init_points,
        num_points=num_points,
        omit_points=omit_points,
        h=dt,
        bounds=None,
        return_status=False,
    )

    if target_dim not in [0, 1, 2]:
        print("targei_dim wrong!")

        exit()

    data = solution[:, target_dim]

    data = np.array(data)

    freq = np.fft.fftfreq(len(data), dt)
    fft_values = np.fft.fft(data)

    power_spectrum = np.abs(fft_values) ** 2

    positive_freqs = freq[freq >= 0]
    positive_power_spectrum = power_spectrum[freq >= 0]

    if plot_show:

        plt.figure(figsize=(8, 2))
        plt.plot(positive_freqs, positive_power_spectrum)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectrum")
        plt.title("Power Spectrum of ODE Solution")
        plt.show()

    if plot_save_dir != None:
        plt.figure(figsize=(8, 2))
        plt.plot(positive_freqs, positive_power_spectrum)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectrum")
        plt.title("Power Spectrum of ODE Solution")
        plt.savefig(plot_save_dir + ".png")

    return positive_freqs, positive_power_spectrum


"""
For verify
"""


def r_simplify(ode, grids, fit_intercept=True):
    vec = ode2vec(ode, grids)
    ode_simplified = vec2ode(
        vec,
        grids,
        variables=["S", "E", "I"],
        name=f"{ode.name}_simplified",
        max_order=ode.max_order,
        extra_term_dicts={},
        fit_intercept=fit_intercept,
    )
    # ode_simplified = vec2ode(vec, grids, variables=seirs.variables, name=f'{ode.name}_simplified', max_order=ode.max_order, extra_term_dicts={}, fit_intercept=fit_intercept)
    return ode_simplified


# ode_shifted_list.append(seirs)


def calc_dist(coef1, coef2, normalization=True):
    assert coef1.shape == coef2.shape
    if normalization:
        return np.sqrt(np.mean(np.square((coef1 - coef2) / (coef2))))
    else:
        return np.sqrt(np.mean(np.square(coef1 - coef2)))


def calc_corr(coef1, coef2):
    assert coef1.shape == coef2.shape
    corr = np.corrcoef(coef1.flatten(), coef2.flatten())[0, 1]
    return corr


def exp_vec(
    vec,
    init_point=None,
    num_points=20000,
    omit_points=1000,
    pad_width=0,
    h=0.0001,
    return_status=False,
    epsilon=EPSILON,
):
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
            print(f"Point {i} Out of Bounds!")
            status = f"Out of Bounds!"
            break
        if np.sum(np.abs(points[-1] - points[-2])) < epsilon:
            print(
                f"Value Converged at Point {i} at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})"
            )
            status = f"Converged at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})"
            break
        if np.abs(points[-1]).max() > INFINITY:
            print(f"Value Exploded at Point {i}!")
            status = f"Exploded!"
            break
    else:
        status = "Normal"

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


def exp_vec_direct(
    vec,
    grids,
    init_point=None,
    num_points=20000,
    omit_points=1000,
    h=0.0001,
    return_status=False,
    use_bounds=True,
    local_tor=500,
    epsilon=EPSILON,
):
    n_dimension = vec.shape[-1]
    if init_point is None:
        init_point = np.random.normal(size=(1, n_dimension))
        for _ in range(10):
            init_point = R_K_direct(vec, grids, init_point[0], h=h)
    else:
        init_point = np.array(init_point).reshape((1, -1))
    points = [init_point]

    min_interval = 2 * np.min(np.diff(grids))
    bounds = np.array(
        [
            [grids[0].min() - min_interval, grids[0].max() + min_interval],
            [grids[1].min() - min_interval, grids[1].max() + min_interval],
            [grids[2].min() - min_interval, grids[2].max() + min_interval],
        ]
    )
    for i in range(1, num_points):
        try:
            points.append(R_K_direct(vec, grids, points[-1][0], h=h))
        except ValueError:
            print(f"Point {i} Out of Bounds!")
            status = f"Out of Bounds!"
            break
        if (
            points[-1][0, 0] < bounds[0, 0]
            or points[-1][0, 0] > bounds[0, 1]
            or points[-1][0, 1] < bounds[1, 0]
            or points[-1][0, 1] > bounds[1, 1]
            or points[-1][0, 2] < bounds[2, 0]
            or points[-1][0, 2] > bounds[2, 1]
        ):
            if use_bounds:
                print(f"Point {i} Out of Bounds!")
                status = f"Out of Bounds!"
                break
            else:
                print(f"Warning: Point {i} Out of Bounds!")
        if np.sum(np.abs(points[-1] - points[-2])) < epsilon:
            print(
                f"Value Converged at Point {i} at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})"
            )
            status = f"Converged at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})"
            break
        if np.abs(points[-1]).max() > INFINITY:
            print(f"Value Exploded at Point {i}!")
            status = f"Exploded!"
            break
        if i > local_tor:
            grid_points = set(
                [find_nearest_point(grids, points[-i][0]) for i in range(1, local_tor)]
            )
            if len(grid_points) < 5:
                points = points[: -local_tor + 1]
                print(
                    f"Value Converged at Point {len(points)} at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})"
                )
                status = f"Converged at ({points[-1][0, 0]:.3f}, {points[-1][0, 1]:.3f}, {points[-1][0, 2]:.3f})"
                break
    else:
        status = "Normal"

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
