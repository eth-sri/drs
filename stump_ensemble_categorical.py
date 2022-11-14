'''
- models a stump ensemble that considers categorical variables and supports l0 certification
'''

import numpy as np



class StumpEnsembleCategorical():
    '''models a stump ensemble supporting categorical features'''
    
    def __init__(self):
        self.predictions = {}
        
    def train(self, X_train, y_train, categorical_indices, w_train=None, lr=1.0):
        '''trains a classifier for every stump independently'''
        
        self.predictions = {}
        if w_train is None:
            w_train = np.ones(len(X_train))
        
        for cat_idx in categorical_indices:
            predictions_current = {}
            cat_num = max(X_train[:, cat_idx]) + 1
            for i in range(cat_num):
                flag = X_train[:, cat_idx] == i
                predictions_current[i] = (int(((np.sum(y_train[flag]*w_train[flag]) / np.sum(w_train[flag])) >= 0.5)*2)-1)*(lr/2.0)+0.5
                
            self.predictions[cat_idx] = predictions_current
            
    def predict_soft_(self, X):
        '''soft prediction'''
        pred = 0.0
        for i, cat in enumerate(self.predictions.keys()):
            if int(X[cat]) in self.predictions[cat]:
                pred += self.predictions[cat][int(X[cat])]
            else:
                pred += 0.5 # not in training data
            
        return pred, len(self.predictions.keys())
    
    def get_perturbations_(self, X, y):
        '''returns score difference through perturbations, worst are first'''
        perturbations = []
        for i, cat in enumerate(self.predictions.keys()):
            perturbation = 0.0
            if int(X[cat]) in self.predictions[cat]:
                pred = self.predictions[cat][int(X[cat])]
            else:
                pred = 0.5
            for j in self.predictions[cat].values():
                if y == 1:
                    perturbation = min(perturbation, -pred + j)
                elif y == 0:
                    perturbation = max(perturbation, -pred + j)
            perturbations.append(perturbation)
        if y == 0:
            perturbations = sorted(perturbations, reverse=True)
        else:
            perturbations = sorted(perturbations)
        return perturbations
                    
    def predict(self, X_test):
        '''hard prediction'''
        y_pred = []
        for k in range(len(X_test)):
            pred, normalizer = self.predict_soft_(X_test[k])
            if pred/normalizer > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
    
    def certify(self, X_test, y_test, radius):
        '''returns whether sampels are certifiably robust at l0 radius'''
        y_cert = []
        worst_case_predictions, normalizer = self.worst_case_predictions(X_test, y_test, radius)
        for k in range(len(X_test)):
            certifiable = 0
            if y_test[k] == 1:
                if worst_case_predictions[k] / normalizer > 0.5:
                    certifiable = 1
            else:
                if worst_case_predictions[k] / normalizer <= 0.5:
                    certifiable = 1
            y_cert.append(certifiable)
        return y_cert
    
    def worst_case_predictions(self, X_test, y_test, radius):
        '''considers the worst case predictions at l0 radius'''
        y_worst_case_predictions = []
        for k in range(len(X_test)):
            pred, normalizer = self.predict_soft_(X_test[k])
            worst_case_prediction = pred
            worst_perturbations = self.get_perturbations_(X_test[k], y_test[k])
            for r in range(radius):
                worst_case_prediction += worst_perturbations[r]
            y_worst_case_predictions.append(worst_case_prediction)
        return y_worst_case_predictions, normalizer
        