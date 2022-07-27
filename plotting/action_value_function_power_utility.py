import numpy as np

from plotting.action_value_function_baseline import plot_q_value_surfaces
from utils.opt_q_values import get_opt_q_values_power_ut


def plot_optimal_q_value_surface_pow_ut(times, wealths, risky_asset_allocations, mu, r, sigma, dt, T,title="Optimal Q-value surface"):
    X, Y = np.meshgrid(risky_asset_allocations, wealths)
    Zs = [get_opt_q_values_power_ut(Y, X, t, T, dt, r, mu, sigma) for t in times]
    title = title + " at time t={}"
    plot_q_value_surfaces(X, Y, Zs, times, title)

    return Zs