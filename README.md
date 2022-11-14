# (De-)Randomized Smoothing for Decision Stump Ensembles <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

This codebase contains code and models to reproduce the experiments of our NeurIPS'22 paper [(De-)Randomized Smoothing for Decision Stump Ensembles](https://www.sri.inf.ethz.ch/publications/horvath2022derand).

## Overview

We propose DRS, a (De-)Randomized Smoothing approach for robustness certification of decision stump ensembles. We equip DRS with various training methods, and also support joint certificates of numerical and categorical features. Empirically, we obtain significant state-of-the-art improvements for certified L_p robustness of tree-based models.

Scripts to reproduce our results can be found in `./scripts`. While all training should be deterministic and can be completed quickly, we also provide pre-trained models in `./models`.

## Getting Started

Please set up a conda environment as follows:

```
conda create --name drs_env python=3.8
conda activate drs_env
pip install -r requirements.txt
```

To install pytorch 1.8.0 and torchvision 0.9.0, the following commands can be used (depending on the installed CUDA version that can be checked, e.g., by running `nvidia-smi`):
```
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

If you are not using the most recent GPU drivers, a compatible cudatoolkit version can be found  [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

Finally, please install `tensorflow` via:

```
conda install -c conda-forge tensorflow==2.4.1
```

## Examples

This codebase is structured into certification and training using four different methods: ensembles of independently MLE optimal stumps, RobTreeBoost, RobAdaBoost, and joint certification of categorical and numerical variables. Here, we provide the key examples for each of these methods.

More scripts, e.g. for ablation studies, can be found in `./scripts`. Once run, the corresponding results are saved in the `./experiments` folder.

### Independently Trained Stump Ensembles

The independently trained stump ensemble pipeline can be run via the scripts `scripts/independent_l1.sh` or `scripts/independent_l2.sh` for l1 and l2 norm certification, respectively. 
For each dataset, we first train a stump ensemble via `train.py`, then certify it via `certify.py` and finally create LaTex and Markdown tables via `analyze.py`, saved in the `./experiments` folder.

For cross-validated results, the analogous scripts are `scripts/independent_l1_cv.sh` and `scripts/independent_l2_cv.sh`.

### RobTreeBoost

The key results for robust gradient boosting of decision stump ensembles can be reproduced via `scripts/robtreeboost_l1.sh` and `scripts/robtreeboost_l2.sh` for l1 and l2 norm certification, respectively. They include training and evaluation for all gradient boosted models reported in the paper. 


### RobAdaBoost

The key results for robust adaptive boosting of decision stump ensembles can be reproduced via `scripts/robadaboost_l1.sh` and `scripts/robadaboost_l2.sh` for l1 and l2 norm certification, respectively. They include training the models, evaluating them, and saving the main results in tables in the `./experiments` folder. 

### Joint Certification

The key results for joint certification of numerical and categorical variables can be reproduced via `scripts/joint_certification_l1.sh` and  `scripts/joint_certification_l2.sh` for l1 and l2 norm certification, respectively. It includes training the models, evaluating them, and saving the main results in tables in the `./experiments` folder.

For cross-validated results, the analogous scripts are `scripts/joint_certification_cv_l1.sh` and `scripts/scripts/joint_certification_cv_l2.sh`.

## Contributors

- Miklós Z. Horváth
- [Mark Niklas Müller](https://www.sri.inf.ethz.ch/people/mark)
- [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

## Citation

If you find this work useful for your research, please cite it as:

```
@inproceedings{
    horvath2022derandomized,
    author = {Miklós Z. Horváth and Mark Niklas M{\"{u}}ller and Marc Fischer and Martin Vechev},
    title = {(De-)Randomized Smoothing for Decision Stump Ensembles},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2022},
    url = {https://openreview.net/forum?id=IbBHnPyjkco},
}
```
