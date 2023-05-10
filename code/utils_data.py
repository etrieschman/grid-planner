import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


PATH_HOME = Path(os.getcwd()).parent
PATH_MODELS = PATH_HOME / 'models'
PATH_DATA = PATH_HOME / 'data'
PATH_RESULTS = PATH_HOME / 'results'


def make_fake_data(N, T, n_nodes, n_series):
   # define initial datapoint
    Sigma = np.random.normal(size=(n_nodes*n_series, n_nodes*n_series))*20
    Sigma = Sigma.T @ Sigma
    mu = np.random.normal(size=n_nodes*n_series)*10
    beta = np.random.randint(1,99,n_nodes*n_series)/100

    # assume moving average system
    X0 = np.random.multivariate_normal(mean=mu, cov=Sigma, size=N)
    X = [X0]
    for t in tqdm(range(T-1)):
        X += [X[-1]*beta + np.random.normal(size=n_nodes*n_series)]
    X = np.array(X)
    X = X.reshape(N, n_series, T, n_nodes)
    print('Returning dataset with shape:\t', X.shape)
    return X



def train_val_test_dataset(dataset, subset, test_split, val_split, seed):
    
    if subset is None:
      idxs = list(range(len(dataset)))
    else:
      idxs = subset
    
    # split off test data first
    trainval_idx, test_idx = train_test_split(idxs, test_size=test_split, random_state=seed)
    # split remaining train/val indexes
    train_idx, val_idx = train_test_split(
      list(range(len(trainval_idx))), test_size=val_split, random_state=seed)
    # map those indices back onto original dataset
    train_idx = [trainval_idx[i] for i in train_idx]
    val_idx = [trainval_idx[j] for j in val_idx]
    datasets = {
      'all': Subset(dataset, idxs),
      'train': Subset(dataset, train_idx),
      'validate': Subset(dataset, val_idx),
      'test': Subset(dataset, test_idx)
      }
    return datasets


def make_dataloaders(
  dataset, subset, test_split=0.10, val_split=0.25, 
  batch_size=32, seed=None):
  
  N = len(dataset)
  # split into train/val
  datasets = train_val_test_dataset(dataset, subset, test_split, val_split, seed)

  # make dataloaders
  dataloaders = {
    k:DataLoader(datasets[k], batch_size=batch_size, shuffle=True) 
    for k in ['all', 'train','validate','test']
    }

  # make dictionary of sizes
  ds_sizes = {
    k:len(datasets[k]) for k in ['all', 'train','validate','test']
  }

  return dataloaders, ds_sizes



  