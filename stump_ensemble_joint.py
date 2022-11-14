'''
- models a stump ensemble that jointly supports categorical and numerical variables, and supports joint robustness certification
'''

import numpy as np
from scipy.stats import norm



class StumpEnsembleJoint():
    '''a model which supports categorical and numerical features modularly'''
    
    EPS = 1e-15
    
    def __init__(self, se_numerical, se_categorical, indices_numerical):
        '''initiates data for pretrained models'''
        self.se_numerical = se_numerical
        self.se_categorical = se_categorical
        self.discretization = se_numerical.discretization
        self.indices_numerical = indices_numerical
    
    def get_cdf(self, value=0, sigma=0.25, mode='gaussian'):
        '''returns cumulative density function'''
        if mode == 'gaussian': # for l2 certification of numerical features
            cdf = lambda x: norm.cdf((x - value) / sigma)
        elif mode == 'uniform': # for l1 certification of numerical features
            cdf = lambda x: np.clip((x-value) / (2*sigma) + 0.5, 0, 1)
        else:
            raise Exception('mode not supported')
        return cdf
        
    def certify_deterministic_joint(self, x, y, l0_radius, sigma, mode, use_categorical = True):
        '''does joint certification'''
        if use_categorical: # use categorical features; default
            dp = np.zeros((len(self.se_numerical.indices)+len(self.se_categorical.predictions))*self.discretization+1)
       
            # shift through categorical variables for pdf
            categorical_worst_case = self.se_categorical.worst_case_predictions(x.reshape(1, -1), np.array([[y]]), l0_radius)[0]
            shift = round(categorical_worst_case[0] * self.discretization)
            dp[shift] = 1.0
        else: # do not use categorical variables
            dp = np.zeros(len(self.se_numerical.indices)*self.discretization+1)
            dp[0]=1.0
        
        # shifts numerical variables for pdf
        for i in range(len(self.se_numerical.indices)):
            current_idx = self.indices_numerical[self.se_numerical.indices[i]]
            cdf = self.get_cdf(self.se_numerical.values[i], sigma=sigma, mode=mode)
            p_right = cdf(x[current_idx])
            p_left = 1 - p_right
            new_dp = np.zeros_like(dp)
            shift_left = self.se_numerical.predictions_left[i]
            shift_right = self.se_numerical.predictions_right[i]
            if shift_left > 0:
                new_dp[shift_left:] += dp[:-shift_left] * p_left
            else:
                new_dp[:] += dp[:] * p_left
            if shift_right > 0:
                new_dp[shift_right:] += dp[:-shift_right] * p_right
            else:
                new_dp[:] += dp[:] * p_right
            dp = new_dp
            
        # computing certifiable radius
        middle_index = int(len(dp)/2)
        p0Bar = np.sum(dp[:middle_index+1])
        p1Bar = 1-p0Bar
        if y == 1:
            pABar = p1Bar
        else:
            pABar = p0Bar
        if pABar < 0.5:
            return False, 0, pABar
        else:
            pABar = min(pABar, 1-self.EPS)
            if mode == 'gaussian':
                radius = sigma * norm.ppf(pABar)
            elif mode == 'uniform':
                radius = 2 * sigma * (pABar - 0.5)
            else:
                raise Exception('mode not supported')
            return True, radius, pABar
        