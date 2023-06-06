import numpy as np
import scipy.stats as sps
import os

PATH_HOME = os.path.dirname(os.getcwd())

class dict_list(dict):
    '''helper class that inherits type dict; used for dictionaries of lists'''
    def __init__(self):
        super(dict_list, self).__init__()
        self.__dict__ = self
        
    def add(self, k, v):
        if k in self:
            self[k].append(v)
        else:
            self[k] = [v]  


def add_stats(results, f, fpred):
    '''helper function to add distribution statistics to a dictionary'''
    # distribution stats
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    for p in percentiles:
        results.add(f'pctl_{p}', np.percentile(fpred, p))
    results.add('mean', np.mean(fpred))
    results.add('std', np.std(fpred))
    
    # comparison stats
    results.add('test_t', sps.ttest_ind(f, fpred).pvalue)
    if not np.all(f - fpred == 0):
        results.add('test_wilcoxon', sps.wilcoxon(f, fpred).pvalue)
    results.add('test_levene', sps.levene(f, fpred).pvalue)
    results.add('test_komolgorov', sps.ks_2samp(f, fpred).pvalue)
    
    return results


def get_stats(f, fpreds):
    results = dict_list()
    for k in fpreds.keys():
        results = add_stats(results, f, fpreds[k])
    return results