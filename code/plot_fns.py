import matplotlib.pyplot as plt

def set_plt_settings():
    plt.rcParams.update({'font.size': 18})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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

