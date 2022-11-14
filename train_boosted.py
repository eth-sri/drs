import argparse
import data
import numpy as np
from pathlib import Path
import pickle
from time import time
import torch

from boosted_stump_ensemble import StumpEnsemble

parser = argparse.ArgumentParser(description='train stump ensembles')
parser.add_argument('dataset', type=str, help='dataset to consider')
parser.add_argument('mode', type=str, help='kind of noise for training')  # gaussian, uniform and default are supported
parser.add_argument('sigma', type=float, help='noise to consider for training')
parser.add_argument('--bin_size', type=float, default=0.01, help='bin size to consider for optimizing the thresholds v')
parser.add_argument('--balanced', type=int, default=0, help='whether dataset should be balanced')
parser.add_argument('--min_impurity', type=float, default=-0.693147, help='minimum impurity for choosing indices')
parser.add_argument('--use_noisy_samples', type=int, default=0, help='whether to add random noise for training')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--boosting', type=str, default=None, choices=["standard", "cdf_inverse"])
parser.add_argument('--warmup', type=int, default=None, help='Number of random features to use for warmup')
parser.add_argument("--boosting_percentile", type=float, default=None)
parser.add_argument("--max_feature_reuse", type=int, default=5)
parser.add_argument("--max_boosting_classifiers", type=int, default=100)
parser.add_argument("--boosting_lr", type=float, default=1)

parser.add_argument("--cert_sigma", type=float, default=1.0)
parser.add_argument("--second_order_approx", type=int, default=20,
                    help="when to start using second order approximation for inverse cdf")
parser.add_argument("--approx_kernel_width", type=int, default=40, help="radius of the smoothing kernel")
parser.add_argument("--approx_kernel_sigma", type=float, default=0.05, help="std of the smoothing kernel")
parser.add_argument("--discretization_steps", type=int, default=100, help="weight discretization steps")
parser.add_argument("--subsample_training_cdf", type=int, default=-1,
                    help="compute CDF only on subsample during training")

args = parser.parse_args()

np.random.seed(args.seed)

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.tensor([1], device=device)

    # loading the dataset
    X_train, y_train, X_test, y_test, eps = data.all_datasets_dict[args.dataset]()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    print('dataset loaded with shapes: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    if args.use_noisy_samples == 1:
        X_train = X_train + np.random.randn(*X_train.shape) * args.sigma


    # assigns weights to samples
    w_train = np.ones(len(X_train))
    if args.balanced == 1:
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        if n_class_0 > n_class_1:
            w_train[y_train == 1] *= float(n_class_0) / n_class_1
        elif n_class_1 > n_class_0:
            w_train[y_train == 0] *= float(n_class_1) / n_class_0

    start = time()

    # initializing stump ensemble
    stump_ensemble = StumpEnsemble(mode=args.mode, sigma=args.sigma, bin_size=args.bin_size)

    # training stump ensemble
    if args.warmup or args.boosting is None:
        stump_ensemble.train_independent_stumps(X_train, y_train, w_train, min_impurity=args.min_impurity,
                                                random_features=args.warmup)
        stump_ensemble.discretize(args.discretization_steps)
        warmup_end = time()
        print(f"warmup time: {warmup_end - start:.4f}")
        cert_radii_pre_boosting = stump_ensemble.certify(X_test, y_test, noise_sigma=args.cert_sigma, device=device)
        cert_end = time()

        print('Pre Boosting ACR', np.mean(cert_radii_pre_boosting))
        for r in np.arange(0, 3.05, 0.05):
            acc_cert = (cert_radii_pre_boosting > r).mean()
            print(f"Pre Boosting cert acc at R = {r:.2f} is {acc_cert:.4f}")
            if acc_cert == 0: break
    else:
        cert_end = start
        warmup_end = start
        cert_radii_pre_boosting = None
        stump_ensemble.discretize(args.discretization_steps)



    if args.boosting is not None:
        if args.boosting == "standard":
            stump_ensemble.train_boosting(X_train, y_train, w_train, lr=args.boosting_lr,
                                          relevant_percentile=args.boosting_percentile,
                                          max_feature_reuse=args.max_feature_reuse,
                                          max_iterations=args.max_boosting_classifiers, device=device)
        train_end = time()
        print(f"train time: {train_end - cert_end + warmup_end - start:.4f}")

        cert_radii_post_boosting = stump_ensemble.certify(X_test, y_test, noise_sigma=args.cert_sigma, device=device)
        cert_end = time()
        print(f"cert time: {cert_end - train_end:.4f}")

        print('Post Boosting ACR', np.mean(cert_radii_post_boosting))
        for r in np.arange(0, 3.05, 0.05):
            acc_cert = (cert_radii_post_boosting > r).mean()
            print(
                f"Post Boosting: cert acc at R = {r:.2f} : {acc_cert:.4f}; delta: {-1. if cert_radii_pre_boosting is None else acc_cert - (cert_radii_pre_boosting > r).mean():.4f}")
            if acc_cert == 0: break

    # saving stump ensemble
    output_path = f'models/{args.dataset}'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    file_name = f'/data_{args.dataset}_mode_{args.mode}_sigma_{args.sigma}_{args.balanced}_{args.use_noisy_samples}_{args.seed}_{int(time())}.pkl'
    print(f"Model saved to {file_name}")
    with open(output_path + file_name, 'wb') as f:
        pickle.dump(stump_ensemble, f)
