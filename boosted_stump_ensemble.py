import numpy as np
import torch
from scipy.stats import entropy, norm
import scipy.stats
from tqdm import tqdm


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

    def discretize(self, steps=100, normalize=False):
        if normalize:
            self.predictions_left = (self.predictions_left + 1) / 2.0
            self.predictions_right = (self.predictions_right + 1) / 2.0
            self.offset = (self.offset + 1) / 2.0

        if self.discretization is not None and steps != self.discretization:
            print(f"WARNING! Discretizing and already discretized model with different step size.")
        d = 1 if self.discretization is None else 1 / self.discretization
        if len(self.indices) > 0:
            assert self.predictions_left.min() >= 0.0 and self.predictions_left.max() * d <= 1.0
            assert self.predictions_right.min() >= 0.0 and self.predictions_right.max() * d <= 1.0
        self.predictions_right = np.round(self.predictions_right * steps * d).astype(int)
        self.predictions_left = np.round(self.predictions_left * steps * d).astype(int)
        self.offset = np.round(self.offset * steps * d).astype(int)
        self.discretization = steps
        print(f"Discretized model with n={steps} steps.")

    def discretize_weight(self, x):
        assert self.discretization is not None
        return np.round(x * self.discretization).astype(int)

    def discretize_boosting(self, steps=100):
        '''discretization if predictions are in [-1,1] originally'''
        self.discretize(steps=steps, normalize=True)

    def fix_old_ensemble(self):
        self.offset = 0
        self.offset_count = 0
        self.predictions_right = self.predictions_right[:, 1]
        self.predictions_left = self.predictions_left[:, 1]
        self.num_stumps = (self.indices > -1).sum()

    def prune(self):
        ambivalent_idx = (self.predictions_left == self.predictions_right)
        if ambivalent_idx.sum() > 0:
            self.offset = self.offset + self.predictions_right[ambivalent_idx].sum()
            self.offset_count += ambivalent_idx.sum()
        self.indices = self.indices[~ambivalent_idx]
        self.predictions_right = self.predictions_right[~ambivalent_idx]
        self.predictions_left = self.predictions_left[~ambivalent_idx]
        self.values = self.values[~ambivalent_idx]
        self.impurities = self.impurities[~ambivalent_idx]

    def add_capacity(self, n):
        existing_stump_count = self.indices.shape[0]
        self.indices = np.concatenate([self.indices, -np.ones(n, dtype=self.indices.dtype)], 0)
        self.values = np.concatenate([self.values, -np.ones(n, dtype=self.values.dtype)], 0)
        self.impurities = np.concatenate([self.impurities, -np.ones(n, dtype=self.impurities.dtype)], 0)
        self.predictions_left = np.concatenate([self.predictions_left, -np.ones(n, dtype=self.predictions_left.dtype)],
                                               0)
        self.predictions_right = np.concatenate(
            [self.predictions_right, -np.ones(n, dtype=self.predictions_right.dtype)], 0)
        return existing_stump_count

    ################
    ### training ###
    ################

    def train_independent_stumps(self, X_train, y_train, w_train, min_impurity=-0.693, random_features=None):
        '''trains a stump ensemble, optimizing a stump for each feature independently'''

        # training one stump for each index
        n_features = X_train.shape[1]
        initial_idx = self.add_capacity(n_features)

        if random_features is None or random_features >= n_features:
            features = range(n_features)
        else:
            features = np.random.choice(n_features, random_features, replace=False)

        for index in features:
            stump_idx = initial_idx + index
            best_value, best_impurity, prediction_left, prediction_right = self.get_best_split(X_train, y_train,
                                                                                               w_train, index)
            if best_value is None: continue
            print(
                f'index {index}, best_value {best_value}, best_impurity {best_impurity}, predictions: {prediction_left}, {prediction_right}')
            self.indices[stump_idx] = index
            self.values[stump_idx] = best_value
            self.impurities[stump_idx] = best_impurity
            self.predictions_left[stump_idx] = prediction_left
            self.predictions_right[stump_idx] = prediction_right
            self.num_stumps += 1
            y_pred = self.predict(X_train)
            print(f"natural train accuracy: {np.equal(y_train, y_pred >= 0.5).mean()}")

        self.filter_stumps(self.indices > -1)
        self.filter_stumps(self.impurities > min_impurity)

    def train_boosting(self, X_train, y_train, w_train, discretizations=100,
                       max_iterations=1000, relevant_percentile=None, lr=1.00, max_feature_reuse=3, device="cpu"):
        '''trains stump ensemble in a boosted fashion (direct implementation of algorithm 6 from Friedman'''
        assert self.discretization is not None

        # training one stump for each index, sequentially based on previous errors
        dp, dp_min, _ = self.get_ensemble_pdf_fast(X_train, device=device)

        y_train = torch.tensor(y_train, device=device)

        # initial values for gradient boosting
        if self.num_stumps == 0:
            y_pred = 0.5 * torch.ones(y_train.shape, dtype=torch.int, device=device)
        elif relevant_percentile is None:
            y_pred = torch.tensor(self.predict(X_train), device=device)
        else:
            p_1 = (dp_min + self.get_percentile(dp, 1 - relevant_percentile)) / (discretizations * self.num_stumps)
            p_0 = (dp_min + self.get_percentile(dp, relevant_percentile)) / (discretizations * self.num_stumps)
            y_pred = torch.where(y_train == 1, p_1, p_0)

        y_current = ((y_train - y_pred + 1) / 2).to("cpu").numpy()

        n_features = X_train.shape[1]
        initial_idx = self.add_capacity(max_iterations)

        for iteration in range(max_iterations):
            stump_idx = initial_idx + iteration

            best_index, best_value, best_impurity, prediction_left, prediction_right = None, None, -np.inf, None, None

            pbar = tqdm(range(n_features))
            for index in pbar:
                if (self.indices == index).sum() > max_feature_reuse: continue
                if n_features == 28 ** 2:
                    n_skip = 3
                    if index < 28 * n_skip or index > (28 - n_skip) * 28: continue
                    if index % 28 < n_skip or index % 28 > (28 - n_skip): continue

                candidate_value, candidate_impurity, candidate_prediction_left, candidate_prediction_right = self.get_best_split(
                    X_train, y_current, w_train, index, "mse", lr, best_impurity)
                if candidate_impurity > best_impurity:
                    best_index, best_value, best_impurity, prediction_left, prediction_right = index, candidate_value, candidate_impurity, candidate_prediction_left, candidate_prediction_right
                    pbar.set_description(
                        f"Using {self.num_stumps} stumps and considering index {best_index} and split {best_value:.3f} with best impurity {best_impurity:.4f}")

            if best_index is None:
                print(f"No stump found improving impurity. Terminating")
                break

            prediction_right = self.discretize_weight(prediction_right)
            prediction_left = self.discretize_weight(prediction_left)
            print(f'index {best_index}, best_value {best_value:.3f}, best_impurity {best_impurity:5f}, predictions: {prediction_left}, {prediction_right}')

            # saving things
            self.indices[stump_idx] = best_index
            self.values[stump_idx] = best_value
            self.impurities[stump_idx] = best_impurity
            self.predictions_left[stump_idx] = prediction_left
            self.predictions_right[stump_idx] = prediction_right
            self.num_stumps += 1

            dp, dp_min, _ = self.get_ensemble_pdf_fast(X_train, device=device)

            if relevant_percentile is None:
                y_pred = torch.tensor(self.predict(X_train), device=device)
                y_current = ((y_train - y_pred) / 2 + 0.5).to("cpu").numpy()
                train_acc = torch.eq(y_train, y_pred > 0.5).float().mean().item()
                print(f"natural train accuracy: {train_acc}")
            else:
                p_1 = (dp_min + self.get_percentile(dp, 1 - relevant_percentile)) / (discretizations * self.num_stumps)
                p_0 = (dp_min + self.get_percentile(dp, relevant_percentile)) / (discretizations * self.num_stumps)
                y_percentile = torch.where(y_train == 1, p_1, p_0)
                y_current = ((y_train - y_percentile) / 2 + 0.5).to("cpu").numpy()
                train_acc = torch.eq(y_train, y_percentile > 0.5).float().mean().item()
                print(f"percentile {relevant_percentile:.2f} train accuracy: {train_acc}")

        # updating values
        self.filter_stumps(self.indices > -1)

    def train_boosting_cdf(self, X_train, y_train, w_train, relevant_percentile, max_iterations=1000,
                           max_feature_reuse=3, approx_k_sigma=0.05, approx_k_width=40, approx_second_order=20,
                           subsample_train=-1):
        assert self.discretization is not None
        initial_idx = self.add_capacity(max_iterations)

        for iteration in range(max_iterations):
            stump_idx = initial_idx + iteration
            self.get_best_stump_cdf_inv(X_train, y_train, w_train, relevant_percentile, stump_idx, max_feature_reuse,
                                        approx_k_sigma, approx_k_width, approx_second_order, subsample_train)
            y_pred = self.predict(X_train)
            print(f"natural train accuracy: {np.equal(y_train, y_pred >= 0.5).mean()}")

        # updating values
        self.filter_stumps(self.indices > -1)

    def get_percentile(self, pdf, percentile):
        if isinstance(pdf, torch.Tensor):
            cdf = torch.cumsum(pdf, 1)
            answer = (cdf > percentile).int().argmax(1)
        else:
            cdf = np.cumsum(pdf, 1)
            answer = (cdf > percentile).argmax(1)
        return answer

    #################################
    ### filtering out weak stumps ###
    #################################

    def filter_stumps(self, idx):
        '''filters out stumps which do not satisfy idx'''
        self.indices = self.indices[idx]
        self.values = self.values[idx]
        self.impurities = self.impurities[idx]
        self.predictions_right = self.predictions_right[idx]
        self.predictions_left = self.predictions_left[idx]
        self.num_stumps -= sum(self.indices > -1) - sum(idx)

    ##########################
    ### finding best split ###
    ##########################

    def get_best_split(self, X_train, y_train, w_train, index, loss="entropy_impurity", lr=None, best_impurity=-np.inf):
        '''computes the best split new_split at the given index'''
        if loss == "entropy_impurity":
            loss_FN = self.entropy_impurity
        elif loss == "mse":
            loss_FN = self.mse_impurity
        else:
            assert False, f"loss function {loss} unknown"

        best_value, prediction_left, prediction_right = None, None, None
        l_extension = 1 * self.sigma if self.mode == "uniform" else 0
        r_extension = 1 * self.sigma if self.mode == "uniform" else 0
        for new_split in np.arange(np.min(X_train[:, index]) - l_extension - self.bin_size / 2,
                                   np.max(X_train[:, index]) + self.bin_size + r_extension, self.bin_size):
            p, g, idx_new = self.get_probabilities_training(X_train, index, new_split)
            w = p * np.expand_dims(w_train / w_train.mean(), 0)
            impurity = loss_FN(y_train, w)
            if impurity > best_impurity:
                prediction_left, prediction_right = self.compute_optimal_leaf_predictions(y_train, w_train, p, idx_new,
                                                                                          lr)
                best_value, best_impurity = new_split, impurity
        return best_value, best_impurity, prediction_left, prediction_right

    ###############################
    ### computing probabilities ###
    ###############################

    def get_feature_cdf(self, value=0, x=None, noise_sigma=None, device="cpu"):
        '''returns cumulative density function'''
        if isinstance(x, torch.Tensor):
            x = x.to(device)
            value = torch.tensor(value, device=device)
            norm = torch.distributions.normal.Normal(0, 1)
            clip = torch.clip
        else:
            clip = np.clip
            norm = scipy.stats.norm

        if noise_sigma is None:
            noise_sigma = self.sigma
        if x is None:
            if self.mode == 'gaussian':
                cdf = lambda x: norm.cdf((x - value) / noise_sigma)
            elif self.mode == 'uniform':
                cdf = lambda x: clip((x - value) / (2 * noise_sigma) + 0.5, 0, 1)
            elif self.mode == 'default':
                cdf = lambda x: (x >= value) * 1.0
            else:
                raise Exception('mode not supported')
            return cdf
        else:
            if self.mode == 'gaussian':
                return norm.cdf((x - value) / noise_sigma)
            elif self.mode == 'uniform':
                return clip((x - value) / (2 * noise_sigma) + 0.5, 0, 1)
            elif self.mode == 'default':
                return (x >= value) * 1.0
            else:
                raise Exception('mode not supported')

    def get_ensemble_pdf_fast(self, X, omited_idx=None, noise_sigma=None, device="cpu"):
        assert self.discretization is not None, "Can only get pdf of a discretized stump ensemble! Call `self.discretize()`."
        n_samples = X.shape[0]
        valid_idxs = self.indices > -1
        used_indices = set(self.indices[valid_idxs])
        if omited_idx is not None:
            used_indices = used_indices - set(omited_idx)
        used_indices = np.array(list(used_indices))

        dp = torch.ones((n_samples, 1), device=device)
        dp_min = self.offset
        dp_max = self.offset
        n_stumps_considered = 0
        width_dp = 1
        X = torch.tensor(X, device=device)

        if not len(used_indices) == 0:
            index_idx = np.isin(self.indices, used_indices)
            index_set = self.indices[index_idx]
            p_rights = self.get_feature_cdf(self.values[index_idx], X[:, index_set], noise_sigma=noise_sigma,
                                            device=device)

            for idx in used_indices:
                n_feature_used = sum(self.indices[valid_idxs] == idx)
                n_stumps_considered += n_feature_used
                p, g = self.get_probabilities_cert(X, idx, p_rights, index_set, device=device)
                min_g = np.min(g)
                max_g = np.max(g)
                width_dp += max_g - min_g
                g_offset = g - min_g
                new_dp = torch.zeros((n_samples, width_dp), device=device)
                dp_min += min_g
                dp_max += max_g
                for pi, gi in zip(p, g_offset):
                    new_dp[:, gi:gi + dp.shape[1]] += dp * pi.unsqueeze(1)
                dp = new_dp
        cert_idx = int(0.5 * self.discretization) * (n_stumps_considered + self.offset_count) - dp_min
        return dp, dp_min, cert_idx

    def get_probabilities_cert(self, X, index, p_right, index_set, device="cpu"):
        '''computes the stump's decision probabuilities'''

        dtype = self.predictions_right.dtype

        splits = self.values[self.indices == index]
        split_order = np.argsort(splits)

        g = np.zeros((len(splits) + 1), dtype=dtype)
        g[1:] += self.predictions_right[self.indices == index][split_order].cumsum()
        g[:-1] += self.predictions_left[self.indices == index][split_order][::-1].cumsum()[::-1]

        cdf_i = torch.zeros((X.shape[0], len(splits) + 2), device=device)
        cdf_i[:, 1:-1] = 1. - p_right[:, index_set == index][:, split_order]
        cdf_i[:, -1] = 1
        p = cdf_i[:, 1:] - cdf_i[:, :-1]
        return p.permute(1, 0), g

    def get_probabilities_training(self, X, index, new_split=None):
        '''computes the stump's decision probabuilities'''

        dtype = self.predictions_right.dtype

        if new_split is None:
            splits = np.array([])
            new_split_g = np.array([], dtype=dtype)
        else:
            splits = np.array([new_split])
            new_split_g = np.array([0], dtype=dtype)

        old_splits = self.values[self.indices == index]
        if len(old_splits) > 0:
            old_splits = np.concatenate([old_splits, splits], 0)
            split_order = np.argsort(old_splits)

            g_l = np.concatenate(
                [np.concatenate([self.predictions_left[self.indices == index], new_split_g], 0)[split_order],
                 np.array([0], dtype=dtype)], 0)[::-1]
            g_r = np.concatenate([np.array([0], dtype=dtype),
                                  np.concatenate([self.predictions_right[self.indices == index], new_split_g], 0)[
                                      split_order]], 0)
            g = g_l.cumsum()[::-1] + g_r.cumsum()
            splits = old_splits[split_order]
            idx_new = split_order[-1] + 1
        else:
            g = np.zeros((2,), dtype=dtype)
            idx_new = 1
        cdf_i = [np.zeros(X.shape[0])] + [1. - self.get_feature_cdf(split, X[:, index]) for split in splits] + [
            np.ones(X.shape[0])]
        p = np.array([cdf_i[i + 1] - cdf_i[i] for i in range(len(cdf_i) - 1)])
        return p, g, idx_new

    ################
    ### criteria ###
    ################

    def entropy_impurity(self, y_train, w):
        '''computes the entropy impurity for given labels and weights/probabilities'''

        counts = np.stack([((1 - y_train) * w).sum(1), (y_train * w).sum(1)], 1)
        weighted_entropy = [entropy(counts[i]) * w[i].sum() for i in range(w.shape[0]) if w[i].sum() > 0]
        entropy_impurity = -sum(weighted_entropy) / w.sum()

        return entropy_impurity

    def mse_impurity(self, y_train, w):
        '''computes the mse impurity for given labels and weights/probabilities'''

        if any(w.sum(1) == 0):
            mse_impurity = 1000000.0
        else:
            mu = np.expand_dims((y_train * w).mean(1) / w.mean(1), 1)
            # mu = np.expand_dims((y_train * w).mean(1), 1)
            mse_impurity = ((y_train - mu) ** 2 * w).sum() / w.sum()

        return -mse_impurity

    ####################################
    ### assign predictions to leaves ###
    ####################################

    def compute_optimal_leaf_predictions(self, y_train, w_train, p, idx_new=1, boosting_lr=None):
        '''computes MLE predictions for leaves'''
        D = 1e-6
        w = p * np.expand_dims(w_train / w_train.mean(), 0)

        if boosting_lr is not None:
            nominator_left = np.sum((y_train - 0.5) * 2 * w[:idx_new])
            nominator_right = np.sum((y_train - 0.5) * 2 * w[idx_new:])

            denominator_helper = lambda y: abs(y * 2 - 1) * (1 - abs(y * 2 - 1))
            denominator_left = np.sum((D + denominator_helper(y_train)) * w[:idx_new])
            denominator_right = np.sum((D + denominator_helper(y_train)) * w[idx_new:])
            prediction_left = np.clip(nominator_left / denominator_left * boosting_lr / 4 + 0.5, 0, 1)
            prediction_right = np.clip(nominator_right / denominator_right * boosting_lr / 4 + 0.5, 0, 1)
        else:
            nominator_left = np.sum(y_train * w[:idx_new])
            nominator_right = np.sum(y_train * w[idx_new:])
            denominator_left = w[:idx_new].sum()
            denominator_right = w[idx_new:].sum()

            prediction_left = np.clip(nominator_left / denominator_left, 0, 1)
            prediction_right = np.clip(nominator_right / denominator_right, 0, 1)

        return prediction_left, prediction_right

    ###############
    ### predict ###
    ###############

    def predict(self, X):
        '''computes predictions on test samples'''
        d = 1. if self.discretization is None else 1. / self.discretization
        valid_indices = self.indices > -1
        if len(self.indices) == 0:
            predictions = np.zeros(X.shape[0], dtype=self.predictions_right.dtype)
        else:
            predictions = np.where(X[:, self.indices] < np.expand_dims(self.values, 0), self.predictions_left * d,
                                   self.predictions_right * d)[:, valid_indices].sum(1)
        if self.num_stumps > 0:
            predictions = (predictions + self.offset * d) / self.num_stumps
        return predictions

    def certify(self, X, y, noise_sigma=None, device="cpu"):
        pdf, pdf_min, cert_idx = self.get_ensemble_pdf_fast(X, noise_sigma=noise_sigma, device=device)
        p_bar = torch.sum(pdf[:, cert_idx + 1:], 1)
        y_pred = (p_bar > 0.5)
        correct = torch.eq(y_pred.to(dtype=torch.int), torch.tensor(y, dtype=torch.int, device=device)).to(
            "cpu").numpy()
        pABar = torch.where(y_pred, p_bar, 1 - p_bar).to("cpu").numpy()
        radius = self.get_radius_from_p_bar(pABar, noise_sigma)
        certified_radii = np.where(correct, radius, 0.)
        return certified_radii

    def get_radius_from_p_bar(self, pBar, noise_sigma=None):
        D = 1e-8
        pBar = np.minimum(pBar.astype(np.float64), 1 - D)
        if noise_sigma is None:
            noise_sigma = self.sigma

        if self.mode == 'gaussian':
            radius = noise_sigma * norm.ppf(pBar)
        elif self.mode == 'uniform':
            radius = 2 * noise_sigma * (pBar - 0.5)
        else:
            raise Exception('noise mode not supported')
        return radius
