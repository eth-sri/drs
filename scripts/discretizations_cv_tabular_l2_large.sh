python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_2500_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 2500; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_2500_0/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_2500_0 certified_radii gaussian 4.0 --device cuda --current_fold 0 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_2500_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 2500; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_2500_1/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_2500_1 certified_radii gaussian 4.0 --device cuda --current_fold 1 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_2500_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 2500; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_2500_2/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_2500_2 certified_radii gaussian 4.0 --device cuda --current_fold 2 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_2500_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 2500; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_2500_3/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_2500_3 certified_radii gaussian 4.0 --device cuda --current_fold 3 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_2500_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 2500; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_2500_4/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_2500_4 certified_radii gaussian 4.0 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/breast_cancer_l2_2500/table 4 ./experiments/discretizations_cv/breast_cancer_l2_2500_0/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_2500_1/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_2500_2/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_2500_3/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_2500_4/certified_radii; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_5000_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 5000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_5000_0/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_5000_0 certified_radii gaussian 4.0 --device cuda --current_fold 0 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_5000_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 5000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_5000_1/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_5000_1 certified_radii gaussian 4.0 --device cuda --current_fold 1 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_5000_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 5000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_5000_2/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_5000_2 certified_radii gaussian 4.0 --device cuda --current_fold 2 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_5000_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 5000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_5000_3/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_5000_3 certified_radii gaussian 4.0 --device cuda --current_fold 3 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_5000_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 5000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_5000_4/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_5000_4 certified_radii gaussian 4.0 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/breast_cancer_l2_5000/table 4 ./experiments/discretizations_cv/breast_cancer_l2_5000_0/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_5000_1/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_5000_2/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_5000_3/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_5000_4/certified_radii; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_10000_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 10000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_10000_0/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_10000_0 certified_radii gaussian 4.0 --device cuda --current_fold 0 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_10000_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 10000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_10000_1/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_10000_1 certified_radii gaussian 4.0 --device cuda --current_fold 1 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_10000_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 10000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_10000_2/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_10000_2 certified_radii gaussian 4.0 --device cuda --current_fold 2 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_10000_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 10000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_10000_3/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_10000_3 certified_radii gaussian 4.0 --device cuda --current_fold 3 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_10000_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 10000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_10000_4/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_10000_4 certified_radii gaussian 4.0 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/breast_cancer_l2_10000/table 4 ./experiments/discretizations_cv/breast_cancer_l2_10000_0/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_10000_1/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_10000_2/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_10000_3/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_10000_4/certified_radii; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_25000_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 25000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_25000_0/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_25000_0 certified_radii gaussian 4.0 --device cuda --current_fold 0 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_25000_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 25000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_25000_1/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_25000_1 certified_radii gaussian 4.0 --device cuda --current_fold 1 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_25000_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 25000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_25000_2/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_25000_2 certified_radii gaussian 4.0 --device cuda --current_fold 2 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_25000_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 25000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_25000_3/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_25000_3 certified_radii gaussian 4.0 --device cuda --current_fold 3 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_25000_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 25000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_25000_4/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_25000_4 certified_radii gaussian 4.0 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/breast_cancer_l2_25000/table 4 ./experiments/discretizations_cv/breast_cancer_l2_25000_0/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_25000_1/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_25000_2/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_25000_3/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_25000_4/certified_radii; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_50000_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 50000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_50000_0/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_50000_0 certified_radii gaussian 4.0 --device cuda --current_fold 0 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_50000_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 50000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_50000_1/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_50000_1 certified_radii gaussian 4.0 --device cuda --current_fold 1 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_50000_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 50000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_50000_2/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_50000_2 certified_radii gaussian 4.0 --device cuda --current_fold 2 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_50000_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 50000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_50000_3/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_50000_3 certified_radii gaussian 4.0 --device cuda --current_fold 3 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_50000_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 50000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_50000_4/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_50000_4 certified_radii gaussian 4.0 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/breast_cancer_l2_50000/table 4 ./experiments/discretizations_cv/breast_cancer_l2_50000_0/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_50000_1/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_50000_2/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_50000_3/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_50000_4/certified_radii; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_100000_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 100000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_100000_0/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_100000_0 certified_radii gaussian 4.0 --device cuda --current_fold 0 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_100000_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 100000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_100000_1/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_100000_1 certified_radii gaussian 4.0 --device cuda --current_fold 1 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_100000_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 100000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_100000_2/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_100000_2 certified_radii gaussian 4.0 --device cuda --current_fold 2 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_100000_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 100000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_100000_3/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_100000_3 certified_radii gaussian 4.0 --device cuda --current_fold 3 --n_splits 5; \
python train.py breast_cancer gaussian 4.0 --output_path ./models/discretizations_cv/breast_cancer_l2_100000_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 100000; \
python certify.py breast_cancer ./models/discretizations_cv/breast_cancer_l2_100000_4/stump_ensemble.pkl ./experiments/discretizations_cv//breast_cancer_l2_100000_4 certified_radii gaussian 4.0 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/breast_cancer_l2_100000/table 4 ./experiments/discretizations_cv/breast_cancer_l2_100000_0/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_100000_1/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_100000_2/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_100000_3/certified_radii ./experiments/discretizations_cv/breast_cancer_l2_100000_4/certified_radii; \