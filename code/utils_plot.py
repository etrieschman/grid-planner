import numpy as np
import matplotlib.pyplot as plt
from utils import PATH_HOME

FIG_SCALE = 1

def set_plt_settings():
    # plt.rcParams.update({'font.size': 18})
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_dist(f, fpred, bins=30, save_file=None):
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
    ax[0].set_xlabel('Samples, sorted by cost')
    ax[1].set_xlabel('Empirical distribution')
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].legend()

    if save_file is not None:
        plt.savefig(PATH_HOME + 'results/' + save_file + '.png', pad_inches=0.5)
    plt.show()


def plot_boxplots(f, fpreds, xlim=None, save_file=None):
    n0, N = np.array(list(fpreds.keys())).min(), np.array(list(fpreds.keys())).max()
    flier_dict = {'markersize':1, 'alpha':0.5}
    box_dict = {'alpha':0.5}
    whisker_dict = {'alpha':0.5}
    fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=[len(fpreds)/3,1], 
                           figsize=FIG_SCALE*(3,4))
    fig.tight_layout()
    ax[0].boxplot(list(reversed(fpreds.values())), vert=False, labels=list(reversed(fpreds.keys())),
                flierprops=flier_dict, boxprops=box_dict, whiskerprops=whisker_dict)
    ax[1].boxplot(f, vert=False, labels=['True model'], patch_artist=True,
                flierprops=flier_dict, boxprops=box_dict, whiskerprops=whisker_dict)
    ax[0].set_ylabel('Surrogate model, fit with N samples')
    ax[1].set_xlabel('Normalized cost')
    ax[0].set_yticks(range(n0, N, int(np.ceil((N - n0)/5))))
    ax[0].set_yticklabels(range(N, n0, -int(np.ceil((N - n0)/5))))
    if xlim is not None:
        ax[0].set_xlim(xlim)

    if save_file is not None:
        plt.savefig(PATH_HOME + 'results/' + save_file + '.png', pad_inches=0.5)
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

