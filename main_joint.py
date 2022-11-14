'''
- main file for joint certification of numerical and categorical variables
'''

import argparse
import numpy as np
import data_joint
from time import time
import pickle
import os

from stump_ensemble import StumpEnsemble
from stump_ensemble_categorical import StumpEnsembleCategorical
from stump_ensemble_joint import StumpEnsembleJoint

parser = argparse.ArgumentParser(description='certify a stump ensemble jointly supporting categorical and numerical variables')
parser.add_argument('dataset', type=str, help='dataset to consider for experiment')
parser.add_argument('mode', type=str, help='noise mode to consider for smoothing (either gaussian or uniform')
parser.add_argument('sigma', type=float, help='magnitude to consider for noise')
parser.add_argument('data_output_path', type=str, help='path where data should be saved to')
parser.add_argument('model_output_path', type=str, help='path where model should be saved to')
parser.add_argument('--balanced', type=int, default=1, help='whether training should be balanced')
parser.add_argument('--lr', type=float, default=0.25, help='learning rate for categorical training (effect of each variable)')
parser.add_argument('--discretizations', type=int, default=100, help='discretizations to consider')
args = parser.parse_args()

def compute(dataset, mode, sigma, balanced, lr, l0_radii, discretizations, thresholds, data_output_path, model_output_path, current_fold, n_splits):
    
    # loading the data
    X_train, y_train, X_test, y_test, n_classes, cat_idx, num_idx = data_joint.all_datasets_dict[dataset](current_fold, n_splits)
    
    # current suffix (used as an id)
    if n_splits == None:
        suffix = '-1_-1'
    else:
        suffix = f'{n_splits}_{current_fold}'
                    
    # preparing output folder
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    # training categorical data
    w_train = np.ones(len(X_train))
    if balanced:
        scaling = np.mean(y_train == 0) / np.mean(y_train == 1)
        w_train[y_train == 1] = w_train[y_train == 1] * scaling
    se_categorical = StumpEnsembleCategorical()
    se_categorical.train(X_train, y_train, cat_idx, w_train, lr=lr)
    
    # certifyting categorical data
    file_name = f'{data_output_path}/main_results_categorical_{suffix}'
    f = open(file_name, 'w')
    print("l0 Radius & CA & BCA", file=f)
    for l0_radius in l0_radii:
        y_certified = se_categorical.certify(X_test, y_test, l0_radius)
        flag0 = y_test == 0
        flag1 = y_test == 1
        a0 = np.mean(np.array(y_certified)[flag0])
        a1 = np.mean(np.array(y_certified)[flag1])
        bca = (a0+a1)/2.0
        ca = np.mean(y_certified)
        print(l0_radius, end=' ', file=f)
        print("& {:.1f}".format(100.0*ca), end=' ', file=f)
        print("& {:.1f}".format(100.0*bca), file=f, flush=True)
    f.close()
    
    if dataset == 'mushroom':
        return
    
    # numerical features only
    X_train_num = np.array(X_train[:, num_idx], dtype=float)
    se_numeric = StumpEnsemble(mode=mode, sigma=sigma)
    se_numeric.train(X_train_num, y_train, balanced=balanced==1, n_jobs=1)
    se_numeric.discretize(discretizations)
        
    # joint model
    se_joint = StumpEnsembleJoint(se_numeric, se_categorical, num_idx)
        
    # certifying numerical data only
    certified_radii = []
    for i in range(len(X_test)):
        correct, radius, pABar = se_joint.certify_deterministic_joint(X_test[i], y_test[i], l0_radius=-1, sigma=sigma, mode=mode, use_categorical=False)
        certified_radii.append(radius)
    certified_radii = np.array(certified_radii)
    np.save(f'{data_output_path}/certified_radii_n_{suffix}.npy', certified_radii)
        
    # joint certification of categorical and numerical data
    for l0_radius in l0_radii:
        certified_radii = []
        for i in range(len(X_test)):
            correct, radius, pABar = se_joint.certify_deterministic_joint(X_test[i], y_test[i], l0_radius, sigma, mode, use_categorical=True)
            certified_radii.append(radius)
        certified_radii = np.array(certified_radii)
        np.save(f'{data_output_path}/certified_radii_nc{l0_radius}_{suffix}.npy', certified_radii)
        
    # saving the joint model
    with open(f'{model_output_path}/stump_ensemble_joint_model_{suffix}', 'wb') as f:
        pickle.dump(se_joint, f)
    
def analyze(data_output_path, thresholds, dataset, current_fold, n_splits, l0_radii):
    
    if dataset == 'mushroom':
        return
    
    # loading the data
    X_train, y_train, X_test, y_test, n_classes, cat_idx, num_idx = data_joint.all_datasets_dict[dataset](current_fold, n_splits)
    
    # current suffix (used as an id)
    if n_splits == None:
        suffix = '-1_-1'
    else:
        suffix = f'{n_splits}_{current_fold}'
    
    # main results - balanced
    f = open(f'{data_output_path}/main_results_balanced_{suffix}', 'w')
    for i in thresholds:
        print("& {:.2f}".format(i), end=' ', file=f)
    print(file=f)
    names = ['n', 'nc0', 'nc1', 'nc2', 'nc3']
    if len(l0_radii)==3:
        names = ['n', 'nc0', 'nc1', 'nc2']
    for name in names:
        certified_radii = np.load(f'{data_output_path}/certified_radii_{name}_{suffix}.npy')
        print(name, end=' ', file=f)
        for i in thresholds:
            a0 = np.mean((certified_radii[y_test == 0] > i)*1.0)
            a1 = np.mean((certified_radii[y_test == 1] > i)*1.0)
            a = np.mean((certified_radii > i)*1.0)
            bca = 100 * (a0+a1)/2.0
            print("& {:.1f}".format(bca), end=' ', file=f)
        print(file=f, flush=True)
    f.close()
        
    # main results - imbalanced
    f = open(f'{data_output_path}/main_results_imbalanced_{suffix}', 'w')
    for i in thresholds:
        print("& {:.2f}".format(i), end=' ', file=f)
    print(file=f)
    names = ['n', 'nc0', 'nc1', 'nc2', 'nc3']
    if len(l0_radii)==3:
        names = ['n', 'nc0', 'nc1', 'nc2']
    for name in names:
        certified_radii = np.load(f'{data_output_path}/certified_radii_{name}_{suffix}.npy')
        print(name, end=' ', file=f)
        for i in thresholds:
            ca = 100 * np.mean((certified_radii > i)*1.0)
            print("& {:.1f}".format(ca), end=' ', file=f)
        print(file=f, flush=True)
    f.close()
    
def analyze_cv(data_output_path, thresholds, dataset, l0_radii):
    '''
    summarizes 5-fold cross-validation results into one table with mean and standard deviation
    '''
    
    # main results - categorical
    f =  open(f'{data_output_path}/main_results_categorical_cv', 'w')
    for i in ['CA', 'BCA']:
        print(f"& {i}", end=' ', file=f)
    print(file=f)
    datas = []
    for i in range(5):
        file_path = f'{data_output_path}/main_results_categorical_5_{i}'
        f_current = open(file_path, 'r')
        lines = f_current.readlines()
        if len(l0_radii) > 3:
            data = np.array([np.array(lines[i].split(' & ')[1:]).astype(float) for i in range(1, 5)])
        else:
            data = np.array([np.array(lines[i].split(' & ')[1:]).astype(float) for i in range(1, 4)])
        datas.append(data)
    names = ['0', '1', '2', '3']
    if len(l0_radii)==3:
        names = ['0', '1', '2']
    for j1, name in enumerate(names):
        print(name, end=' ', file=f)
        for j2, i in enumerate(['CA', 'BCA']):
            current_values = [datas[k][j1][j2] for k in range(5)]
            mean = np.mean(current_values)
            std = np.std(current_values)
            print("& {:.1f}".format(mean)+"\\textsubscript{$\pm$ " + "{:.1f}".format(std)+"}", end=' ', file=f)
        print(file=f, flush=True)
    f.close()
    
    if dataset == 'mushroom':
        return
    
    # main results - balanced
    f =  open(f'{data_output_path}/main_results_balanced_cv', 'w')
    for i in thresholds:
        print("& {:.2f}".format(i), end=' ', file=f)
    print(file=f)
    datas = []
    for i in range(5):
        file_path = f'{data_output_path}/main_results_balanced_5_{i}'
        f_current = open(file_path, 'r')
        lines = f_current.readlines()
        if len(l0_radii) > 3:
            data = np.array([np.array(lines[i].split(' & ')[1:]).astype(float) for i in range(1, 6)])
        else:
            data = np.array([np.array(lines[i].split(' & ')[1:]).astype(float) for i in range(1, 5)])
        datas.append(data)
    names = ['n', 'nc0', 'nc1', 'nc2', 'nc3']
    if len(l0_radii)==3:
        names = ['n', 'nc0', 'nc1', 'nc2']
    for j1, name in enumerate(names):
        print(name, end=' ', file=f)
        for j2, i in enumerate(thresholds):
            current_values = [datas[k][j1][j2] for k in range(5)]
            bca_mean = np.mean(current_values)
            bca_std = np.std(current_values)
            print("& {:.1f}".format(bca_mean)+"\\textsubscript{$\pm$ " + "{:.1f}".format(bca_std)+"}", end=' ', file=f)
        print(file=f, flush=True)
    f.close()
    
    # main results - imbalanced
    f =  open(f'{data_output_path}/main_results_imbalanced_cv', 'w')
    for i in thresholds:
        print("& {:.2f}".format(i), end=' ', file=f)
    print(file=f)
    datas = []
    for i in range(5):
        file_path = f'{data_output_path}/main_results_imbalanced_5_{i}'
        f_current = open(file_path, 'r')
        lines = f_current.readlines()
        if len(l0_radii) > 3:
            data = np.array([np.array(lines[i].split(' & ')[1:]).astype(float) for i in range(1, 6)])
        else:
            data = np.array([np.array(lines[i].split(' & ')[1:]).astype(float) for i in range(1, 5)])
        datas.append(data)
    names = ['n', 'nc0', 'nc1', 'nc2', 'nc3']
    if len(l0_radii)==3:
        names = ['n', 'nc0', 'nc1', 'nc2']
    for j1, name in enumerate(names):
        print(name, end=' ', file=f)
        for j2, i in enumerate(thresholds):
            current_values = [datas[k][j1][j2] for k in range(5)]
            ca_mean = np.mean(current_values)
            ca_std = np.std(current_values)
            print("& {:.1f}".format(ca_mean)+"\\textsubscript{$\pm$ " + "{:.1f}".format(ca_std)+"}", end=' ', file=f)
        print(file=f, flush=True)
    f.close()
    
    
if __name__ == '__main__':
    
    # some parameters
    l0_radii = np.array([0,1,2,3])
    if args.dataset == 'mammal':
        l0_radii = np.array([0,1,2])
    thresholds = np.arange(0, 1.75, 0.25)
    
    
    # default train/test split (first 70% is train, last 30% is test)
    compute(dataset=args.dataset, mode=args.mode, sigma=args.sigma, balanced=args.balanced, lr=args.lr, l0_radii=l0_radii, discretizations=args.discretizations, thresholds=thresholds, data_output_path=args.data_output_path, model_output_path=args.model_output_path, current_fold=None, n_splits=None)
    analyze(data_output_path=args.data_output_path, thresholds=thresholds, dataset=args.dataset, current_fold=None, n_splits=None, l0_radii=l0_radii)
    
    
    # 5-fold cross validation
    for i in range(5):
        compute(dataset=args.dataset, mode=args.mode, sigma=args.sigma, balanced=args.balanced, lr=args.lr, l0_radii=l0_radii, discretizations=args.discretizations, thresholds=thresholds, data_output_path=args.data_output_path, model_output_path=args.model_output_path, current_fold=i, n_splits=5)
        analyze(data_output_path=args.data_output_path, thresholds=thresholds, dataset=args.dataset, current_fold=i, n_splits=5, l0_radii=l0_radii)
    
    analyze_cv(data_output_path=args.data_output_path, thresholds=thresholds, dataset=args.dataset, l0_radii=l0_radii)
    