'''
- supports deterministic and randomized smoothing for stump ensembles
- based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import datetime
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from time import time
from pathlib import Path
import pickle
import os

import data
from stump_ensemble import StumpEnsemble



parser = argparse.ArgumentParser(description='certify a stump ensemble')
parser.add_argument('dataset', type=str, help='dataset to consider for certification')
parser.add_argument('model_path', type=str, help='path to the model that needs to be loaded')
parser.add_argument('output_path', type=str, help='path to file where output needs to be saved')
parser.add_argument('file_name', type=str, help='name of the file where certification results will be saved')
parser.add_argument('mode', type=str, help='noise mode to consider for smoothing (either gaussian or uniform')
parser.add_argument('sigma', type=float, help='magnitude to consider for noise (sigma for l2, lambda for l1)')
parser.add_argument('--smoothing_mode', default=0, type=int, help='whether to use deterministic (0) or randomized smoothing (1)')
parser.add_argument('--n0', type=int, default=100, help='number of perturbed samples for selecting majority class (for RS)')
parser.add_argument('--n', type=int, default=100000, help='number of perturbed samples for certification (for RS)')
parser.add_argument('--alpha', type=float, default=0.001, help='confidence in certification (for RS)')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility (different runs)')
parser.add_argument('--split', type=str, default='test', help='test or train data for certification')
parser.add_argument('--num_classes', type=int, default=2, help='number of the datasets classes')
parser.add_argument('--device', type=str, default='cpu', help='device for certification')
parser.add_argument('--debug', type=int, default=0, help='debug mode')
parser.add_argument('--test_from', type=int, default=0, help='first sample to test')
parser.add_argument('--test_num', type=int, default=np.inf, help='number of samples to test')
parser.add_argument('--current_fold', type=int, default=None, help='current cross-validation fold')
parser.add_argument('--n_splits', type=int, default=None, help='size of cross validation')
args = parser.parse_args()

ABSTAIN = -1
EPS = 1e-15



def certify(dataset, model_path, output_path, file_name, mode, sigma, smoothing_mode, n0, n, alpha, seed, device):
    '''does certification'''
    
    np.random.seed(seed)
    
    # loading the dataset
    X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[dataset](args.current_fold, args.n_splits)
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    
    # loading the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # creating output file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f = open(output_path+'/'+file_name, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    
    # obtaining cAHats and pABars
    before_time = time()
    if smoothing_mode == 0: # deterministic smoothing
        probabilities = model.smoothed_predict_deterministic(X_test, mode, sigma, device)
        cAHats = np.argmax(probabilities, axis=1)
        pABars = np.choose(cAHats, probabilities.T)
    elif smoothing_mode == 1: # randomized smoothing
        counts_selection = model.smoothed_predict_probabilistic(X_test, mode, sigma, n0)
        cAHats = np.argmax(counts_selection, axis=1)
        counts_estimation = model.smoothed_predict_probabilistic(X_test, mode, sigma, n)
        nAs = np.choose(cAHats, counts_estimation.T)
        pABars = np.array([_lower_confidence_bound(nA, n, alpha) for nA in nAs])
    else:
        raise Exception('smoothing_mode not supported')
    after_time = time()
    time_elapsed_average = str(datetime.timedelta(seconds=(after_time - before_time))/len(X_test))
      
    # computing certified radii and outputting data
    for i in range(len(X_test)):
        cAHat = cAHats[i]
        pABar = pABars[i]
        certified_radius = 0.0
        if pABar < 0.5:
            cAHat = ABSTAIN
        else:
            pABar = min(pABar, 1. - EPS)
            if mode == 'gaussian':
                certified_radius = sigma * norm.ppf(pABar)
            elif mode == 'uniform':
                certified_radius = 2 * sigma * (pABar - 0.5)
            else:
                raise Exception('noise mode not supported')
        correct = 1 if y_test[i] == cAHat else 0
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, y_test[i], cAHat, certified_radius, correct, time_elapsed_average), file=f, flush=True)
    f.close()
            
def _lower_confidence_bound(NA, N, alpha):
    '''returns a lower bound on the underlying probability'''
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]



if __name__ == '__main__':
    
    certify(dataset=args.dataset, model_path=args.model_path, output_path=args.output_path, file_name=args.file_name, mode=args.mode, sigma=args.sigma, smoothing_mode=args.smoothing_mode, n0=args.n0, n=args.n, alpha=args.alpha, seed=args.seed, device=args.device)

    
    