
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from pandas.plotting import autocorrelation_plot
import seaborn as sns

from mcmc_sampling import create_hmc_sampler


def banana_potential_energy_value(state, a=2.15, b=0.75, rho=0.9, ):
    """
    Potential energy of the posterir. This is dependent on the target state, not the momentum.
    It is the negative the posterior-log, and MUST be implemented for each distribution
    """
    x, y = state[:]
    #
    pdf_val = 1 / (2 * (1 - rho**2))
    t1 = x**2 / a**2 + a**2 * (y - b * x**2 / a**2 - b * a**2)**2
    t2 = - 2 * rho * x * (y - b * x**2 / a**2 - b * a**2)
    pdf_val = (t1 + t2) / (2 * (1 - rho**2))
    #
    return pdf_val

def banana_potential_energy_gradient(state, a=2.15, b=0.75, rho=0.9, ):
    """
    Gradient of the Potential energy of the posterir.
    """
    x, y = state.flatten()
    #
    pdf_grad = np.empty(2)
    pdf_grad[:] = 1 / (2 * (1-rho**2))
    #
    t1_x = 2 * x / a**2 + 2 * a**2 * (y - b * x**2 / a**2 - b * a**2) * (-2 * b * x / a**2)
    t1_y = 2 * a**2 * (y - b * x**2 / a**2 - b * a**2)
    t1 = np.array([t1_x, t1_y])
    #
    t2_x = - 2 * rho * y + 6 * b * rho * x**2 / a**2 + 2 * rho * b * a**2

    t2_y = - 2 * rho * x
    t2 = np.array([t2_x, t2_y])
    #
    pdf_grad *= (t1 + t2)
    #
    return pdf_grad

def sample_banana_distribution(sample_size):
    sampler = create_hmc_sampler(
        size=2,
        log_density=lambda x: -banana_potential_energy_value(x),
        # log_density_grad=lambda x: -banana_potential_energy_gradient(x),
    )
    sample = sampler.sample(sample_size=sample_size, initial_state=[0, 0])

    return sample


def evaluate_pdf(x, verpose=False, from_energy=True):
    """
    Evaluate the probability density function at a given value x.
    x: state at which pdf is evaluated
    """
    potential_energy = banana_potential_energy_value(x)
    pdf_val = np.exp(- potential_energy)

    if verpose:
        print('PDF Value (upto scaling cons = %8.7e ' % (pdf_val))
    #
    if isinstance(pdf_val, np.ndarray):
        if pdf_val.size == 1:
            pdf_val = pdf_val[0]
        else:
            raise ValueError("Somethong wrong with the type or dimensions of pdf_val!")
    return pdf_val




def plot_enhancer(fontsize=14, font_weight='bold', usetex=False):
    font_dict = {'weight': font_weight, 'size': fontsize}
    plt.rc('font', **font_dict)
    plt.rc('text', usetex=usetex)


def start_plotting_2d(collected_ensemble,
                      xlim=(-6, 6),
                      ylim=(-2, 11),
                      linewidth=1.0,
                      markersize=3,
                      fontsize=18,
                      keep_plots=False,
                      ):
    """
    """
    print("*** Creating 2D plots ***")

    # Plot resutls:
    actual_ens = np.asarray(collected_ensemble)
    sample_size = np.size(actual_ens, 0)

    chain_state_repository = actual_ens.copy()

    print("\n Constructing plot information... ")

    # plot 1: data + best-fit mixture
    x_min, x_max = xlim
    y_min, y_max = ylim

    x_size = y_size = 200
    x = np.linspace(x_min, x_max, x_size)
    y = np.linspace(y_min, y_max, y_size)
    x, y = np.meshgrid(x, y)

    posterior_pdf_vals = np.zeros((x_size, y_size))
    for i in range(x_size):
        for j in range(y_size):
            state_tmp = np.array([x[i][j], y[i][j]])
            posterior_pdf_vals[i,j] = evaluate_pdf(state_tmp)

    z_min = np.max(posterior_pdf_vals) - 0.01
    z_max = np.max(posterior_pdf_vals) + 0.03

    # mask:
    posterior_pdf_vals_cp = posterior_pdf_vals.copy()

    # Animate the sampler steps:
    fig1, ax = plt.subplots(facecolor='white')
    ax.contour(x, y, posterior_pdf_vals_cp, colors='k')
    CS = ax.contourf(x, y, posterior_pdf_vals_cp, 14, cmap="RdBu_r")
    line, = ax.plot(chain_state_repository[0, 0], chain_state_repository[0, 1], '-r', linewidth=linewidth, markersize=markersize, alpha=0.75)
    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(x))
        return line,
    def animate(frame_no):
        data = chain_state_repository[: frame_no+1, :]
        # ax.clear()
        line.set_xdata(data[:, 0])
        line.set_ydata(data[:, 1])
        line.set_linewidth(linewidth)
        line.set_color('red')
        ax.scatter(data[-1, 0], data[-1, 1], alpha=0.75, s=15)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.set_title("$Iteration:%04d$" % (frame_no+1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    save_count=np.size(chain_state_repository, 0)
    # save_count = 250
    ani = animation.FuncAnimation(fig1, animate, interval=100, save_count=save_count)
    filename = "MCMC_diagnostics.mp4"
    ani.save(filename, dpi=900)
    print(f"Saved plot to {filename}")
    if not keep_plots:
        plt.close(fig1)

    #
    # Plot prior, likelihood, posterior, and histogram
    fig2 = plt.figure(figsize=(16, 5), facecolor='white')
    # plot contuour of the posterior and scatter of the ensemble
    ax1 = fig2.add_subplot(1, 3, 1)
    # ax1.set_xticks(np.arange(x_min, x_max, 2))
    # ax1.set_xticklabels(np.arange(x_min, x_max, 2))
    ax1.set_xlabel("$x$", fontsize=fontsize)
    ax1.set_ylabel("$y$", fontsize=fontsize)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    CS1 = ax1.contour(x, y, posterior_pdf_vals)
    ax1.scatter(actual_ens[:, 0], actual_ens[:, 1], alpha=0.75, s=15)
    ax1.set_aspect('auto')

    #
    ax2 = fig2.add_subplot(1, 3, 2)
    # ax2.set_xticks(np.arange(x_min, x_max, 2))
    # ax2.set_xticklabels(np.arange(x_min, x_max, 2))
    ax2.set_xlabel("$x$", fontsize=fontsize)
    ax2.set_ylabel("$y$", fontsize=fontsize)
    # ax2.set_xlim(x_min, x_max)
    # ax2.set_ylim(y_min, y_max)
    ax2.contour(x, y, posterior_pdf_vals_cp, linewidth=0.5, colors='k')
    CS2 = ax2.contourf(x, y, posterior_pdf_vals_cp, 14, cmap="RdBu_r", alpha=0.85)
    ax2.plot(chain_state_repository[:, 0], chain_state_repository[:, 1], '-ro', linewidth=linewidth, markersize=markersize, alpha=0.15)
    ax2.set_aspect('auto')

    #
    # Plot autocorrelation:
    ax3 = fig2.add_subplot(1, 3, 3)
    autocorrelation_plot(posterior_pdf_vals, ax=ax3)
    ax3.set_aspect('auto')

    # Plot autocorrelation:
    sep = '-'*30

    fig2.subplots_adjust(right=0.8, wspace=0.25)
    #
    filename = "MCMC_diagnostics.pdf"
    fig2.savefig(filename, dpi=900, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"Saved plot to {filename}")
    if not keep_plots:
        plt.close(fig2)


def start_plotting_nd(collected_ensemble, labels=None, keep_plots=False, ):
    """
    """
    print("*** Creating bivariate plots array ***")
    # plt.close('all')
    # Plot resutls:
    actual_ens = np.asarray(collected_ensemble)
    sample_size, nvars = actual_ens.shape

    if labels is None:
        columns = [f'X{i+1}' for i in range(nvars)]
    else:
        columns = [l for l in labels]

    # Create dataframe object and plot
    df = pd.DataFrame(actual_ens, columns=columns, )
    g = sns.pairplot(df, diag_kind="kde")
    g.map_lower(sns.kdeplot, levels=5, color=".4")
    filename = "MCMC_PairPlot.pdf"
    g.savefig(filename, dpi=900, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"Saved plot to {filename}")
    if not keep_plots:
        plt.close(g.figure)

    # start_plotting_2d(sample)
    if nvars == 2:
        start_plotting_2d(sample, keep_plots=keep_plots, )

if __name__ == '__main__':
    sample = sample_banana_distribution(100)
    start_plotting_nd(sample)


