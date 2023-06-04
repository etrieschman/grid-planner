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