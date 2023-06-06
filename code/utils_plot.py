import numpy as np
import matplotlib.pyplot as plt
from utils import PATH_HOME

FIG_SCALE = 1

def set_plt_settings():
    # plt.rcParams.update({'font.size': 18})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_dist(f, fpred, bins, N, save_file=None):
    fig, ax = plt.subplots(ncols=2, sharey=True, width_ratios=[2,1], figsize=FIG_SCALE*(6,3))
    fig.tight_layout()
    # sorted figure
    ax[0].plot(np.sort(f), label='True model')
    ax[0].plot(np.sort(fpred), label='Surrogate model')
    # histogram
    rlow, rhigh = np.minimum(np.min(f), np.min(fpred)), np.maximum(np.max(f), np.max(fpred))
    ax[1].hist(f, alpha=0.5, label='True model', bins=bins, range=(rlow, rhigh), 
             density=True, orientation='horizontal')
    ax[1].hist(fpred, alpha=0.5, label='Surrogate model', bins=bins, range=(rlow, rhigh), 
             density=True, orientation='horizontal')
    # formatting
    ax[0].set_ylabel('Normalized cost')
    ax[0].set_xlabel(f'Samples, sorted by cost (fit with N={N})')
    ax[1].set_xlabel('Empirical distribution')
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].legend()

    if save_file is not None:
        plt.savefig(PATH_HOME + '/results/' + save_file + '.png', bbox_inches='tight')
    plt.show()


def plot_boxplots(f, fpreds, xlim=None, save_file=None):
    n0, N = np.array(list(fpreds.keys())).min(), np.array(list(fpreds.keys())).max()
    flier_dict = {'markersize':1, 'alpha':0.5}
    box_dict = {'alpha':0.5}
    whisker_dict = {'alpha':0.5}
    fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=[len(fpreds)/3,1], 
                           figsize=FIG_SCALE*(4,4))
    fig.tight_layout()
    ax[0].boxplot(list(reversed(fpreds.values())), vert=False, labels=list(reversed(fpreds.keys())),
                flierprops=flier_dict, boxprops=box_dict, whiskerprops=whisker_dict)
    ax[1].boxplot(f, vert=False, labels=['True model'], patch_artist=True,
                flierprops=flier_dict, boxprops=box_dict, whiskerprops=whisker_dict)
    ax[0].set_ylabel('N samples to fit Surrogate model')
    ax[1].set_xlabel('Normalized cost')
    # ax[0].set_yticks(range(0, N-n0+5, 5))
    # ax[0].set_yticklabels(reversed(range(n0, N+5, 5)))
    if xlim is not None:
        ax[0].set_xlim(xlim)

    if save_file is not None:
        plt.savefig(PATH_HOME + '/results/' + save_file + '.png', bbox_inches='tight')
    plt.show()


def plot_stats(results, results_pred, save_file=None):
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=FIG_SCALE*(6,3))

    ax[2].plot(np.abs(np.array(results_pred['pctl_10']) - results['pctl_10'][0]), 
               alpha=0.5, color='black', label='pctl_10')
    ax[2].plot(np.abs(np.array(results_pred['pctl_90']) - results['pctl_90'][0]), 
               alpha=0.5, color='black', linestyle=':', label='pctl_90')
    ax[1].plot(np.abs(np.array(results_pred['std']) - results['std'][0]), 
               alpha=0.5, color='black', label='std')
    ax[0].plot(np.abs(np.array(results_pred['mean']) - results['mean'][0]), 
               alpha=0.5, color='black', label='mean')
    ax[0].plot(np.abs(np.array(results_pred['pctl_50']) - results['pctl_50'][0]), 
               alpha=0.5, color='black', linestyle=':', label='pctl_50')
            
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_ylim((-0.25, 1.75))
    ax[0].set_ylabel('|True - Surrogate statistic|')
    ax[1].set_xlabel('N samples to fit Surrogate model')

    if save_file is not None:
        plt.savefig(PATH_HOME + '/results/' + save_file + '.png', bbox_inches='tight')
    plt.show()

def plot_stats_boot(bootstrap, n0, N, step=1, save_file=None):
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(6,3))

    panels = [['mean', 'pctl_50'], ['std'], ['pctl_10', 'pctl_90']]
    colors = ['black', 'purple']

    for i, panel in enumerate(panels):
        for j, stat in enumerate(panel):
            stat_mean = bootstrap[stat].mean(0)
            stat_error = bootstrap[stat].std(0)*1.96
            ax[i].plot(np.arange(n0, N+step, step), stat_mean, alpha=0.5, color=colors[j], label=stat)
            ax[i].fill_between(np.arange(n0, N+step, step), stat_mean-stat_error, stat_mean+stat_error, color=colors[j], alpha=0.10)
        ax[i].legend()

    ax[0].set_ylim((-0.25, 1.75))
    ax[0].set_ylabel('|True - Surrogate statistic|')
    ax[1].set_xlabel('N samples to fit Surrogate model')

    if save_file is not None:
        plt.savefig(PATH_HOME + '/results/' + save_file + '.png', bbox_inches='tight')
    plt.show()


def plot_accs(accs):
    '''
    acs : a dictionary of accuracy values
    '''
    plt.subplots(figsize=(15, 5))
    for k in accs.keys():
        plt.plot(accs[k], '-o')
    plt.legend(accs.keys(), loc='upper left')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()

