'''
- models an ensemble of stump ensembles that can be trained via adaptive boosting
'''

import numpy as np
from scipy.stats import norm

import data
from stump_ensemble import StumpEnsemble
from multiprocessing import Pool

import time


class EnsembleStumpEnsemble():
    
    EPS = 1e-15
    
    def __init__(self):
        self.stump_ensembles = []
        self.stump_ensembles_weights = []
        self.stump_ensemble_overall = None
    
    def train_adaboost(self, X_train, y_train, n_trees=20, radius_to_certify=0.25, noise_sigma=0.25, noise_mode='gaussian', bin_size=0.01, balanced = True, num_classes = 2, n_jobs = 16, device='cpu'):
        '''trains an ensemble of stump ensembles via adaptive boosting'''
        
        # initializing sample weights
        w_train = np.ones(len(X_train))
        if balanced:
            sum_0 = np.sum(y_train == 0)
            sum_1 = np.sum(y_train == 1)
            if sum_0 > sum_1:
                w_train[y_train == 1] *= float(sum_0) / sum_1
            elif sum_1 > sum_0:
                w_train[y_train == 0] *= float(sum_1) / sum_0
            w_train = w_train / np.sum(w_train)
        
        # adaboost training
        for i in range(n_trees):
            stump_ensemble = StumpEnsemble(mode=noise_mode, sigma=noise_sigma, bin_size=bin_size)
            stump_ensemble.train(X_train, y_train, w_train=w_train*len(X_train), n_jobs=n_jobs)
            stump_ensemble.discretize(steps=100)
            
            if device == 'cpu':
                batch_size = int(1.0 * len(X_train) / n_jobs + 1)
                datasets = [(stump_ensemble, X_train[(batch_size*j):(batch_size * (j+1))], y_train[(batch_size*j):(batch_size * (j+1))], radius_to_certify, noise_sigma, noise_mode, num_classes, device) for j in range(n_jobs)]
                pool = Pool()
                outputs = pool.map(self.get_certifiability_, datasets)
                certifiability = np.concatenate(outputs)
            elif device == 'cuda':
                certifiability = self.get_certifiability_fast(stump_ensemble, X_train, y_train, radius_to_certify, noise_sigma, noise_mode, num_classes, device)
            else:
                raise Exception('device not supported')
            
            eps_i = np.sum(((1.0 - certifiability) * w_train)) / np.sum(w_train)
            alpha_i = np.log((1-eps_i)/eps_i) + np.log(num_classes - 1)
            w_train = w_train * np.exp(alpha_i * (1.0 - certifiability))
            w_train = w_train / np.sum(w_train)
            self.stump_ensembles.append(stump_ensemble)
            self.stump_ensembles_weights.append(alpha_i)     
        
    #############################################################
    ### functions for computing certifiability at fixed radii ###
    #############################################################
    
    def get_certifiability_(self, dataset):
        '''function for parallelism'''
        stump_ensemble, X_train, y_train, radius_to_certify, noise_sigma, noise_mode, num_classes, device = dataset
        return self.get_certifiability_fast(stump_ensemble, X_train, y_train, radius_to_certify, noise_sigma, noise_mode, num_classes, device)
    
    def get_certified_radii_fast_(self, dataset):
        '''function for parallelism'''
        stump_ensemble, X_train, y_train, noise_sigma, noise_mode, num_classes, device = dataset
        return self.get_certified_radii_fast(stump_ensemble, X_train, y_train, noise_sigma, noise_mode, num_classes, device)
    
    def get_certifiability_fast(self, stump_ensemble, X_train, y_train, radius_to_certify, noise_sigma, noise_mode, num_classes, device):
        '''computes whether samples are certifiably correct at a given radius'''
        certified_radii = self.get_certified_radii_fast(stump_ensemble, X_train, y_train, noise_sigma, noise_mode, num_classes, device)
        certifiability = np.array(certified_radii) > radius_to_certify
        return certifiability
    
    def get_certified_radii_fast(self, stump_ensemble, X_train, y_train, noise_sigma, noise_mode, num_classes, device):
        '''computes certified radii'''
        dp, dp_min, middle_index = stump_ensemble.get_pdf(X=X_train, mode=noise_mode, sigma=noise_sigma, device=device)
        certified_radii = []
        for i in range(len(X_train)):
            p0_bar = np.sum(dp[i][:middle_index+1])
            p1_bar = 1.0 - p0_bar
            if p0_bar > p1_bar:
                cAHat = 0
                pABar = p0_bar
            else:
                cAHat = 1
                pABar = p1_bar
            pABar = min(pABar, 1-self.EPS)
            correct = cAHat == y_train[i]
            
            if noise_mode == 'gaussian':
                radius = noise_sigma * norm.ppf(pABar)
            elif noise_mode == 'uniform':
                radius = 2 * noise_sigma * (pABar - 0.5)
            else:
                raise Exception('Noise mode not supported')
            if not correct:
                radius = 0.0
            certified_radii.append(radius)
        return certified_radii
        
    def get_certified_radii_ensemble(self, X_test, y_test, noise_sigma, noise_mode, num_classes, device):
        '''computees certified radii for an ensemble of stump ensembles'''
        certified_radii = []
        if device == 'cpu':
            datasets = [(stump_ensemble, X_test, y_test, noise_sigma, noise_mode, num_classes, device) for stump_ensemble in self.stump_ensembles]
            pool = Pool()
            outputs = pool.map(self.get_certified_radii_fast_, datasets)
        elif device == 'cuda':
            outputs = []
            for stump_ensemble in self.stump_ensembles:
                outputs.append(self.get_certified_radii_fast(stump_ensemble, X_test, y_test, noise_sigma, noise_mode, num_classes, device))
        else:
            raise Exception('device not supported')
        return np.array(outputs)
       