'''
- trains a certifiably robust stump ensemble with support for numerous parameters
'''

import argparse
import data
import numpy as np
from pathlib import Path
import pickle
from stump_ensemble import StumpEnsemble



parser = argparse.ArgumentParser(description='train stump ensembles')
parser.add_argument('dataset', type=str, help='dataset to consider')
parser.add_argument('mode', type=str, help='kind of noise for training (gaussian, uniform or default)')
parser.add_argument('sigma', type=float, help='noise to consider for training')
parser.add_argument('--output_path', default='./', type=str, help='path to output (model)')
parser.add_argument('--model_name', default='model.pkl', type=str, help='name of model')
parser.add_argument('--discretization', type=int, default=100, help='number of discretizations')
parser.add_argument('--bin_size', type=float, default=0.01, help='bin size to consider for optimizing the thresholds v')
parser.add_argument('--balanced', type=int, default=1, help='whether dataset should be balanced')
parser.add_argument('--min_impurity', type=float, default=-0.693147, help='minimum impurity for choosing indices')
parser.add_argument('--use_noisy_samples', type=int, default=0, help='whether to add random noise for training (0: no noise; 1: gaussian noise; 2: uniform noise')
parser.add_argument('--n_jobs', type=int, default=16, help='number of cpus for parallelization')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--current_fold', type=int, default=None, help='current cross-validation fold')
parser.add_argument('--n_splits', type=int, default=None, help='size of cross validation')
args = parser.parse_args()



def train(dataset, mode, sigma, output_path, model_name, discretization, bin_size, balanced, min_impurity, use_noisy_samples, n_jobs, seed):
    
    np.random.seed(seed)
    
    # loading data
    X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[dataset](args.current_fold, args.n_splits)
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    if use_noisy_samples == 1:
        X_train = X_train + np.random.randn(*X_train.shape) * sigma
    elif use_noisy_samples == 2:
        X_train = X_train + (np.random.rand(*X_train.shape)-0.5)*2.0 * sigma
        
    # initializing stump ensemble
    stump_ensemble = StumpEnsemble(mode=mode, sigma=sigma, bin_size=bin_size)
    
    # training the stump ensemble
    stump_ensemble.train(X_train, y_train, balanced=balanced, min_impurity=min_impurity, n_jobs=n_jobs)
    stump_ensemble.discretize(steps = discretization)
    
    # saving the stump ensemble
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(output_path+'/'+model_name, 'wb') as f:
        pickle.dump(stump_ensemble, f)

if __name__ == '__main__':
    
    train(dataset=args.dataset, mode=args.mode, sigma=args.sigma, output_path=args.output_path, model_name=args.model_name, discretization=args.discretization, bin_size=args.bin_size, balanced=args.balanced, min_impurity=args.min_impurity, use_noisy_samples=args.use_noisy_samples, n_jobs=args.n_jobs, seed=args.seed)
    