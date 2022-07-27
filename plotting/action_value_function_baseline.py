import numpy as np
from matplotlib import pyplot as plt

from utils.action import get_action
from utils.learned_q_values import get_learned_q_values
from utils.opt_q_values import get_opt_q_values_log_ut


def plot_wireframe(x, y, z, x_lab, y_lab, z_lab, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color="black")
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_zlabel(z_lab, linespacing=4)
    plt.title(title)
    plt.show()


def plot_q_value_surface(x, y, z, title, x_lab='investment in risky asset'):
    plot_wireframe(x, y, z, x_lab, y_lab='wealth', z_lab='\n' + 'Q-values', title=title)


def plot_learned_q_value_surface(critic, times, wealths, risky_asset_allocations, T):
    x, y = np.meshgrid(risky_asset_allocations, wealths)
    zs = [get_learned_q_values(critic=critic, t=t, X=np.ravel(x), Y=np.ravel(y)).reshape(x.shape) for t in times]
    title = "Learned Q-value surface at time t={}"
    plot_q_value_surfaces(x, y, zs, times, title)
    return zs

def plot_learned_q_value_surface(critic, times, wealths, risky_asset_allocations, T):
    x, y = np.meshgrid(risky_asset_allocations, wealths)
    zs = [get_learned_q_values(critic=critic, t=t, X=np.ravel(x), Y=np.ravel(y)).reshape(x.shape) for t in times]
    title = "Learned Q-value surface at time t={}"
    plot_q_value_surfaces(x, y, zs, times, title)
    return zs

def plot_learned_q_value_surface2(fn, times, wealths, risky_asset_allocations, T):
    x, y = np.meshgrid(risky_asset_allocations, wealths)
    zs = [get_learned_q_values(critic=critic, t=t, X=np.ravel(x), Y=np.ravel(y)).reshape(x.shape) for t in times]
    title = "Learned Q-value surface at time t={}"
    plot_q_value_surfaces(x, y, zs, times, title)
    return zs

def plot_q_value_surfaces(x, y, zs, times, title, x_lab='investment in risky asset'):
    for t, Z in zip(times, zs):
        plot_q_value_surface(x, y, Z, title.format(t), x_lab)


def plot_optimal_q_value_surface(times, wealths, risky_asset_allocations, mu, r, sigma, dt, T):
    X, Y = np.meshgrid(risky_asset_allocations, wealths)
    Zs = [get_opt_q_values_log_ut(np.ravel(Y), np.ravel(X), t, T, dt, r, mu, sigma).reshape(X.shape) for t in times]
    title = "Optimal Q-value surface at time t={}"
    plot_q_value_surfaces(X, Y, Zs, times, title)
    return Zs


def plot_cross_section_of_optimal_Q_function(Q_values, times, t, wealths, wealth_level, risky_asset_allocations):
    idx_t = list(times).index(t)
    idx_v = list(wealths).index(wealth_level)

    plt.plot(risky_asset_allocations, Q_values[idx_t][idx_v])
    plt.xlabel('investment in risky asset')
    plt.ylabel('Q-Value')
    plt.title("Cross-section of optimal Q-value surface at time t={} and wealth v={}".format(t, wealth_level))

    plt.show()


def get_idx(array, element):
    return list(array).index(element)


def plot_cross_sections_learned_vs_optimal_fix_wealth(opt_Qvalues, learned_Qvalues, times, t, wealths, wealth_level,
                                                      risky_asset_allocations):
    idx_t = get_idx(times, t)
    idx_v = get_idx(wealths, wealth_level)

    y1 = opt_Qvalues[idx_t][idx_v]
    y2 = learned_Qvalues[idx_t][idx_v]

    title = "Cross-section of optimal Q-value surface at time t={} and wealth v={}".format(t, wealth_level)

    plot_optimal_vs_learned(risky_asset_allocations, ys=[y1, y2], xlabel='investment in risky asset', ylabel='Q-Value',
                            title=title)


def plot_cross_sections_learned_vs_optimal_fix_alloc(opt_Qvalues, learned_Qvalues, times, t, wealths, alloc,
                                                     risky_asset_allocations):
    idx_t = get_idx(times, t)
    idx_pi = get_idx(risky_asset_allocations, alloc)

    y1 = np.array(opt_Qvalues[idx_t])[:, idx_pi]
    y2 = np.array(learned_Qvalues[idx_t])[:, idx_pi]

    title = "Cross-section of optimal Q-value surface at time t={} and risky allocation ={}".format(t, alloc)

    plot_optimal_vs_learned(wealths, ys=[y1, y2], xlabel='Wealth level', ylabel='Q-Value', title=title)


def plot_optimal_vs_learned(x, ys, xlabel, ylabel, title, ylim=None):
    labels = ["Optimal", "Learned"]
    for i in np.arange(0, len(ys)):
        print(labels[i])
        plt.plot(x, ys[i], label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend()
    plt.title(title)

    plt.show()


def plot_learned_policy_vs_optimal_policy(policy, r, mu, sigma, T, wealths, times):
    y_1 = (mu - r) / (sigma ** 2) * np.ones(wealths.shape)
    for t in times:
        y_2 = get_action(policy, t, wealths).reshape(wealths.shape)
        title = 'Optimal vs. Learned policy (at time t={})'.format(t)
        plot_optimal_vs_learned(x=wealths, ys=[y_1, y_2], title=title, xlabel="wealth",
                                ylabel="risky asset allocation", ylim=[-1, 1])


def plot_learned_cross_section(x, y, xlabel, ylabel, title, ylim=None):
    plt.plot(x, y, label="Learned")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend()
    plt.title(title)

    plt.show()


def plot_cross_section_learned_fix_wealth(learned_Qvalues, times, t, wealths, wealth_level,
                                          risky_asset_allocations):
    idx_t = get_idx(times, t)
    idx_v = get_idx(wealths, wealth_level)

    y = learned_Qvalues[idx_t][idx_v]

    title = "Cross-section of learned Q-value surface at time t={} and wealth v={}".format(t, wealth_level)

    plot_learned_cross_section(risky_asset_allocations, y, xlabel='investment in risky asset', ylabel='Q-Value',
                               title=title)


def plot_cross_section_learned_fix_alloc(learned_Qvalues, times, t, wealths, alloc,
                                         risky_asset_allocations):
    idx_t = get_idx(times, t)
    idx_pi = get_idx(risky_asset_allocations, alloc)

    y = np.array(learned_Qvalues[idx_t])[:, idx_pi]

    title = "Cross-section of learned Q-value surface at time t={} and risky allocation ={}".format(t, alloc)

    plot_learned_cross_section(wealths, y, xlabel='Wealth level', ylabel='Q-Value', title=title)


def plot_function(x_values, f_values, x_lab, y_lab, title):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    plt.title(title)
    ax.plot(x_values, f_values)
    plt.show()
