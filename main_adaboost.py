'''
- a file to run experiments by training ensembles of stump ensembles via adaptive boosting, certifying them and saving the results
'''


import numpy as np
from time import time
import numpy as np
import argparse
import pickle
import os

from adaboost import EnsembleStumpEnsemble
import data
from stump_ensemble import StumpEnsemble



parser = argparse.ArgumentParser(description='Adaptive Boosting for Certified Robustness')
parser.add_argument("--dataset", type=str, default='mnist_1_5', help="which dataset")
parser.add_argument("--n_ensembles", type=int, default=20, help='number of ensembles to consider')
parser.add_argument("--radius_to_certify", type=float, default=0.8, help='radius to consider for certification')
parser.add_argument("--noise_sigma", type=float, default=0.5, help='noise sigma to consider for (De-)Randomized Smoothing')
parser.add_argument("--noise_mode", type=str, default='gaussian', help='noise mode to consider (gaussian for l2 certification and uniform for l1')
parser.add_argument("--bin_size", type=float, default=0.01, help='bin size to consider during stump training')
parser.add_argument("--num_classes", type=int, default=2, help='number of classes (current code supports num_classes = 2)')
parser.add_argument("--balanced", type=int, default=1, help='whether training should be balanced or not')
parser.add_argument("--n_jobs", type=int, default=16, help='number of cpu parallelizations')
parser.add_argument("--data_output_path", type=str, default='./', help='path to where output data should be saved')
parser.add_argument("--model_output_path", type=str, default='./', help='path to where output model should be saved')
parser.add_argument("--device", type=str, default='cpu', help='device to use (cpu or cuda)')
args = parser.parse_args()



def compute(dataset, n_ensembles, radius_to_certify, noise_sigma, noise_mode, bin_size, balanced, num_classes, n_jobs, device, data_output_path, model_output_path):
    '''does computations (trains ensemble of stump ensembles via adaboost and also computes certified radii for each of the individual stump ensembles'''
    
    # loading data
    X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[dataset]()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    print('Dataset loaded successfully.')

    # creating and training model
    adaboost_model = EnsembleStumpEnsemble()
    adaboost_model.train_adaboost(X_train, y_train, n_trees=n_ensembles, radius_to_certify=radius_to_certify, noise_sigma=noise_sigma, noise_mode=noise_mode, bin_size=bin_size, balanced=bool(balanced), num_classes=num_classes, n_jobs=n_jobs, device=device)
    print('Training completed.')
    
    # saving model
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    with open(f'{model_output_path}/adaboost_model.pickle', 'wb') as handle:
        pickle.dump(adaboost_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # computing certified radii for each individual model
    certified_radii = adaboost_model.get_certified_radii_ensemble(X_test, y_test, noise_sigma=args.noise_sigma, noise_mode=args.noise_mode, num_classes=2, device=args.device)
    print('Certified radii computed with shape', certified_radii.shape, '.')
    
    # saving individual certified radii
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
    np.save(f'{data_output_path}/certified_radii_individual.npy', certified_radii)

def compute_certified_radii_ensemble(certified_radii_individual, individual_weights):
    '''computes certified radii over weighted ensemble'''
    certified_radii = []
    n_samples = len(certified_radii_individual[0])
    n_models = len(certified_radii_individual)
    weights_sum = np.sum(np.abs(individual_weights))
    for i in range(n_samples):
        individual_radii = [(certified_radii_individual[j][i], individual_weights[j]) for j in range(n_models)]
        individual_radii = sorted(individual_radii, reverse=True)
        current_sum = 0.0
        current_radius = 100000000.0
        j = 0
        while current_sum <= weights_sum / 2.0:
            if j >= n_models:
                current_radius = 0.0
                break
            current_radius = individual_radii[j][0]
            if individual_radii[j][1] > 0:
                current_sum += individual_radii[j][1]
            j += 1
        certified_radii.append(current_radius)
    return certified_radii
        
def analyze(data_output_path, model_output_path, thresholds, ensemble_sizes):
    '''computes certified radii over whole ensemble and outputs useful data for the paper'''
    
    # loading things
    certified_radii = np.load(f'{data_output_path}/certified_radii_individual.npy')
    with open(f'{model_output_path}/adaboost_model.pickle', 'rb') as f:
        adaboost_model = pickle.load(f)
    
    # computing certified radii over things
    f = open(data_output_path + '/main_results', 'w')
    for i in thresholds:
        print("& {:.2f}".format(i), end=' ', file=f)
    print(file=f)
    for ensemble_size in ensemble_sizes:
        if ensemble_size > len(certified_radii):
            continue
        certified_radii_ensemble = compute_certified_radii_ensemble(certified_radii[:ensemble_size], adaboost_model.stump_ensembles_weights[:ensemble_size])
        np.save(f'{data_output_path}/certified_radii_ensemble_{ensemble_size}.npy', certified_radii_ensemble)
        print(ensemble_size, end=' ', file=f)
        for i in thresholds:
            ca = 100 * np.mean((certified_radii_ensemble > i)*1.0)
            print("& {:.1f}".format(ca), end=' ', file=f)
        print(file=f, flush=True) 
    f.close()
        
    
    
if __name__ == '__main__':
    
    # computations
    compute(dataset=args.dataset, n_ensembles=args.n_ensembles, radius_to_certify=args.radius_to_certify, noise_sigma=args.noise_sigma, noise_mode=args.noise_mode, bin_size=args.bin_size, balanced=args.balanced, num_classes=args.num_classes, n_jobs=args.n_jobs, device=args.device, data_output_path=args.data_output_path, model_output_path=args.model_output_path)
    
    # analysis
    thresholds = np.arange(0, 1.1, 0.1)
    ensemble_sizes = [1, 3, 5, 10, 20]
    analyze(data_output_path=args.data_output_path, model_output_path=args.model_output_path, thresholds=thresholds, ensemble_sizes=ensemble_sizes)

    
