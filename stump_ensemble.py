'''
- models an ensemble of stumps and supports (de-)randomized smoothing
'''

import numpy as np
import torch
from scipy.stats import entropy
from scipy.stats import norm
from multiprocessing import Pool

class StumpEnsemble():
    '''models an ensemble of stumps'''
    
    ######################
    ### initialization ###
    ######################
    
    def __init__(self, mode='gaussian', sigma=0.25, bin_size=0.01, offset=None, dtype=float):
        '''initiates the stump ensemble'''
        self.mode = mode
        self.sigma = sigma
        self.bin_size = bin_size
        self.indices = np.array([], dtype=int)
        self.values = np.array([], dtype=float)
        self.impurities = np.array([], dtype=float)
        self.predictions_right = np.array([], dtype=dtype)
        self.predictions_left = np.array([], dtype=dtype)
        self.discretization = None
        self.offset = 0 if offset is None else offset
        self.offset_count = 0 if offset is None else 1
        self.num_stumps = 0

    ################
    ### training ###
    ################
        
    def train(self, X_train, y_train, balanced = False, min_impurity = -0.693, w_train = None, n_jobs=16, train_mode=0):
        '''trains a stump ensemble, optimizing a stump for each feature independently'''
        
        # considers weights to train samples
        if w_train is None:
            w_train = np.ones(len(X_train))
            if balanced:
                sum_0 = np.sum(y_train == 0)
                sum_1 = np.sum(y_train == 1)
                if sum_0 > sum_1:
                    w_train[y_train == 1] *= float(sum_0) / sum_1
                elif sum_1 > sum_0:
                    w_train[y_train == 0] *= float(sum_1) / sum_0
        
        # training one stump for each index 
        values, indices, impurities, predictions_left, predictions_right = [], [], [], [], []
        batch_size = int(1.0 * len(X_train[0]) / n_jobs + 1)
        datasets = [(X_train, y_train, w_train, [j for j in range(batch_size*i, min(batch_size * (i+1), len(X_train[0])))], train_mode) for i in range(n_jobs)]
        pool = Pool()
        outputs = pool.map(self.train_stumps_wrapper_, datasets)
        index_counter = 0
        for i in range(len(outputs)):
            best_values_c, best_impurities_c, predictions_left_c, predictions_right_c = outputs[i]
            for j in range(len(best_values_c)):
                indices.append(index_counter)
                index_counter += 1
                values.append(best_values_c[j])
                impurities.append(best_impurities_c[j])
                predictions_left.append(predictions_left_c[j])
                predictions_right.append(predictions_right_c[j])
            
        # updating stump ensemble
        self.indices = np.array(indices)
        self.values = np.array(values)
        self.impurities = np.array(impurities)
        self.predictions_left = np.array(predictions_left)
        self.predictions_right = np.array(predictions_right)
        self.filter_stumps(min_impurity)
        
    def train_stumps_wrapper_(self, dataset):
        '''helper function for cpu parallelism'''
        X_train, y_train, w_train, indices,  train_mode = dataset
        return self.train_stumps_(X_train, y_train, w_train, indices,  train_mode)
    
    def train_stumps_(self, X_train, y_train, w_train, indices,  train_mode):
        '''computes best values and corresponding predictions for list of indices'''
        best_values = []
        best_impurities = []
        predictions_left = []
        predictions_right = []
        for index in indices:
            if  train_mode == 0:
                best_value, best_impurity, prediction_left, prediction_right = self.get_best_split(X_train, y_train, w_train, index)
            else:
                raise Exception('train_mode not supported')
            best_values.append(best_value)
            best_impurities.append(best_impurity)
            predictions_left.append(prediction_left)
            predictions_right.append(prediction_right)
        return best_values, best_impurities, predictions_left, predictions_right 
        
    #######################
    ### post-processing ###
    #######################
    
    def discretize(self, steps=100, normalize=False):
        '''discretizes leaf predictions'''
        self.predictions_left = self.predictions_left[:,1]
        self.predictions_right = self.predictions_right[:,1]
        if normalize:
            self.predictions_left = (self.predictions_left + 1) / 2.0
            self.predictions_right = (self.predictions_right + 1) / 2.0
            self.offset = (self.offset + 1) / 2.0

        if len(self.indices)>0:
            assert self.predictions_left.min()>= 0.0 and self.predictions_left.max()<= 1.0
            assert self.predictions_right.min() >= 0.0 and self.predictions_right.max() <= 1.0
        self.predictions_right = np.round(self.predictions_right*steps).astype(int)
        self.predictions_left = np.round(self.predictions_left*steps).astype(int)
        self.offset = np.round(self.offset*steps).astype(int)
        self.discretization = steps
        print(f"Discretized model with n={steps} steps.")
    
    def filter_stumps(self, min_impurity):
        '''filters out stumps which do not have a sufficiently large entropy'''
        sufficiently_large = self.impurities > min_impurity
        self.indices = self.indices[sufficiently_large]
        self.values = self.values[sufficiently_large]
        self.impurities = self.impurities[sufficiently_large]
        self.predictions_right = self.predictions_right[sufficiently_large]
        self.predictions_left = self.predictions_left[sufficiently_large]
        
    def prune(self):
        '''prunes stumps which have same predictions on the left and on the right'''
        ambivalent_idx = (self.predictions_left == self.predictions_right).all(1)
        self.indices = self.indices[~ambivalent_idx]
        self.predictions_right = self.predictions_right[~ambivalent_idx]
        self.predictions_left = self.predictions_left[~ambivalent_idx]
        self.values = self.values[~ambivalent_idx]
        self.impurities = self.impurities[~ambivalent_idx]
    
    ##########################
    ### finding best split ###
    ##########################
    
    def get_best_split(self, X_train, y_train, w_train, index):
        '''computes the best split value at the given index'''
        best_value, best_impurity, prediction_left, prediction_right = -1000000.0, -1000000.0, np.array([0.5, 0.5]), np.array([0.5, 0.5])
        for value in np.arange(np.min(X_train[:, index])-2*self.sigma, np.max(X_train[:, index])+self.bin_size+2*self.sigma, self.bin_size):
            p_left, p_right = self.get_probabilities(X_train, index, value, self.sigma, self.mode)
            w_left = p_left * w_train
            w_right = p_right * w_train
            impurity = self.entropy_impurity(y_train, w_left, w_right)
            if impurity > best_impurity:
                prediction_left, prediction_right = self.get_leaf_predictions(y_train, w_left, w_right)
                best_value, best_impurity = value, impurity
        return best_value, best_impurity, prediction_left, prediction_right
    
    ###################################################
    ### helper function for computing probabilities ###
    ###################################################
    
    def get_cdf(self, value=0, sigma=0.25, mode='gaussian'):
        '''returns cumulative density function'''
        if mode == 'gaussian':
            cdf = lambda x: norm.cdf((x - value) / sigma)
        elif mode == 'uniform':
            cdf = lambda x: np.clip((x-value) / (2*sigma) + 0.5, 0, 1)
        elif mode == 'default':
            cdf = lambda x: (x  >= value) * 1.0
        else:
            raise Exception('mode not supported')
        return cdf
    
    def get_probabilities(self, X_train, index, value, sigma, mode):
        '''computes the stump's decision probabuilities'''
        cdf_right = self.get_cdf(value, sigma, mode)
        p_right = cdf_right(X_train[:, index])
        p_left = 1 - p_right
        return p_left, p_right
    
    #############################################
    ### entropy criterion under randomization ###
    #############################################
    
    def entropy_impurity(self, y_train, w_left, w_right):
        '''computes the entropy impurity for given labels and weights/probabilities'''
        
        counts_left = np.bincount(y_train.astype(int), weights=w_left)
        counts_right = np.bincount(y_train.astype(int), weights=w_right)
        n_left, n_right = np.sum(counts_left), np.sum(counts_right)
        n_all = n_left + n_right
    
        new_entropy = 0.0
        if n_left >= 1:
            new_entropy += entropy(counts_left) * n_left
        if n_right >= 1:
            new_entropy += entropy(counts_right) * n_right
        entropy_impurity = (-new_entropy) / n_all
        
        return entropy_impurity
    
    ####################################
    ### assign predictions to leaves ###
    ####################################
    
    def get_leaf_predictions(self, y_train, w_left, w_right):
        '''computes MLE-optimal predictions for leaves'''
        prediction_left = np.bincount(y_train.astype(int), weights=w_left)
        if np.sum(prediction_left) > 0:
            prediction_left = prediction_left / np.sum(prediction_left)
        prediction_right = np.bincount(y_train.astype(int), weights=w_right)
        if np.sum(prediction_right) > 0:
            prediction_right = prediction_right / np.sum(prediction_right)
        return prediction_left, prediction_right
        
    #######################################
    ### predictions (without smoothing) ###
    #######################################
  
    def predict(self, X):
        '''computes (soft) predictions without smoothing (>0.5 maps to class 1; <= 0.5 to class 0)'''
        predictions = np.zeros(len(X))
        d = 1. if self.discretization is None else 1./self.discretization
        for i in range(len(self.indices)):
            flag_left = X[:, self.indices[i]] < self.values[i]
            predictions[flag_left] += self.predictions_left[i]
            flag_right = X[:, self.indices[i]] >= self.values[i]
            predictions[flag_right] += self.predictions_right[i]
        predictions = predictions / float(len(self.indices))
        predictions = predictions * d
        return predictions
    
    ############################
    ### smoothed predictions ###
    ############################
    
    def smoothed_predict_deterministic(self, X, mode, sigma, device):
        '''determinstic smoothed prediction; returns class probabilities for each sample'''
        dp, dp_min, middle_index = self.get_pdf(X, mode, sigma, device)
        probabilities = []
        for i in range(len(X)):
            p0_bar = np.sum(dp[i, :middle_index+1])
            p1_bar = 1.0 - p0_bar
            probabilities.append([p0_bar, p1_bar])
        probabilities = np.array(probabilities)
        return probabilities
    
    def smoothed_predict_probabilistic(self, X, mode, sigma, n):
        '''probabilistic smoothed prediction; returns class counts for each sample'''
        counts = []
        for x in X:
            count = self.smoothed_predict_probabilistic_sample(x, mode, sigma, n)
            counts.append([n-count[1], count[1]])
        counts = np.array(counts)
        return counts
               
    def smoothed_predict_probabilistic_sample(self, x, mode, sigma, n):
        '''probabilistic smoothed prediction for one sample x; returns counts for both classes'''
        if mode == 'gaussian':
            noise = np.random.randn(*[n, *x.shape])
            X_batch = np.expand_dims(x,0) + noise * sigma
        elif mode == 'uniform':
            noise = (np.random.rand(*[n, *x.shape])-0.5)*2.0
            X_batch = np.expand_dims(x,0) + noise * sigma
        else:
            raise Exception('mode not supported')
        predictions = self.predict(X_batch)
        count_1 = np.sum((predictions > 0.5))
        counts = np.array([n-count_1, count_1])
        return counts
    
    ############################################
    ### fast pdf computation (for smoothing) ###
    ############################################
    
    def get_cdf_torch(self, Xi, value, sigma, mode):
        '''computes cdf for torch'''
        if mode == 'gaussian':
            return torch.distributions.normal.Normal(value, sigma).cdf(Xi)
        elif mode == 'uniform':
            return torch.clip((Xi-value) / (2*sigma) + 0.5, 0.0, 1.0)
        else:
            raise Exception('mode not supported')
    
    def get_pdf(self, X, mode, sigma, device='cpu'):
        '''computes pdf for non-multi-stumps'''
        # preprocessing
        X = torch.tensor(X, device=device)
        n_samples = len(X)
        predictions_min = np.minimum(self.predictions_left,self.predictions_right)
        predictions_max = np.maximum(self.predictions_left,self.predictions_right)
        shifts_left = self.predictions_left - predictions_min
        shifts_right = self.predictions_right - predictions_min
        min_dp = np.sum(predictions_min)
        middle_index = int(0.5 * len(self.indices) * self.discretization)-min_dp
        dp_size = np.sum(shifts_left)+np.sum(shifts_right)+1
    
        # computing dp
        dp = torch.ones((n_samples, 1), device=device, dtype=torch.double) # for faster computation: torch.float 
        width_dp = 1
        for i in range(len(self.indices)):
            width_dp += int(max(shifts_left[i], shifts_right[i]))
            new_dp = torch.zeros((n_samples, width_dp), device=device, dtype=torch.double)
            p_right = self.get_cdf_torch(Xi=X[:, self.indices[i]], value=self.values[i], sigma=sigma, mode=mode)
            p_left = 1.0 - p_right
            if shifts_left[i] == 0 and shifts_right[i] == 0:
                continue
            if shifts_left[i] == 0:
                new_dp[:, :dp.shape[1]] = dp * p_left.reshape(-1, 1)
                new_dp[:, shifts_right[i]:shifts_right[i]+dp.shape[1]] += dp * p_right.reshape(-1, 1)
            elif shifts_right[i] == 0:
                new_dp[:, :dp.shape[1]] = dp * p_right.reshape(-1, 1)
                new_dp[:, shifts_left[i]:shifts_left[i]+dp.shape[1]] += dp * p_left.reshape(-1, 1)
            dp = new_dp
        dp = dp.cpu().detach().numpy()
        return dp, min_dp, middle_index
    
    def get_pdf_multi(self, X, mode, sigma, device):
        '''computes pdf supporting multi-stumps'''
        if mode not in ['gaussian', 'uniform']:
            raise Exception('mode not supported')
        X = torch.tensor(X, device=device)
        
        # compute all the thresholds for each index
        index_thresholds = {}
        for i, index in enumerate(self.indices):
            if index in index_thresholds:
                index_thresholds[index].append((self.values[i], self.predictions_left[i], self.predictions_right[i]))
            else:
                index_thresholds[index] = [(self.values[i], self.predictions_left[i], self.predictions_right[i])]
        for index in index_thresholds.keys():
            index_thresholds[index] = sorted(index_thresholds[index])
        
        # computing probabilities
        n_samples = len(X)
        dp = torch.zeros((n_samples, len(self.indices) * self.discretization + 1), device=device)
        
        dp[:, 0] = 1.0
        new_dp = torch.zeros_like(dp, device=device)
        for index in index_thresholds.keys():
        
            # computing new prediction values
            a = index_thresholds[index]
            new_predictions = []
            new_predictions.append(int(np.sum([a[i][1] for i in range(len(a))])))
            for i in range(len(a)):
                new_predictions.append(new_predictions[-1] - a[i][1] + a[i][2])
        
            # computing probabilities
            new_probabilities = torch.zeros((n_samples, len(a)+1), device=device)
            new_probabilities[:, 0] = 1.0 - self.get_cdf_torch(Xi=X[:, index], value=a[0][0], sigma=sigma, mode=mode)
            previous_probabilities = new_probabilities[:, 0] * 1.0
            for i in range(1, len(a)):
                new_probabilities[:, i] = (1.0-self.get_cdf_torch(Xi=X[:, index], value=a[i][0], sigma=sigma, mode=mode))-previous_probabilities
                previous_probabilities = (1.0-self.get_cdf_torch(Xi=X[:, index], value=a[i][0], sigma=sigma, mode=mode)) * 1.0
            new_probabilities[:, len(a)] = self.get_cdf_torch(Xi=X[:, index], value=a[len(a)-1][0], sigma=sigma, mode=mode)
        
            # updating pdf
            new_dp *= 0.0
            visited = False
            for i in range(len(new_predictions)):
                shift = int(new_predictions[i])
                if shift > 0:
                    new_dp[:, shift:] += dp[:, :-shift] * new_probabilities[:, i].reshape(-1,1)
                else:
                    new_dp[:, :] += dp[:, :] * new_probabilities[:, i].reshape(-1,1)
            dp = new_dp*1.0
            
        middle_index = int(0.5 * len(self.indices) * self.discretization)
        dp = dp.cpu().detach().numpy()
        return dp, 0, middle_index
   