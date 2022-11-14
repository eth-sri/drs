python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 1; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 1; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 1; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 1; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 1; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_1/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_1_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_2_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 2; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_2_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_2_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_2_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 2; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_2_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_2_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_2_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 2; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_2_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_2_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_2_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 2; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_2_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_2_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_2_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 2; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_2_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_2_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_2/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_2_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_2_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_2_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_2_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_2_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_3_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 3; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_3_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_3_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_3_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 3; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_3_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_3_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_3_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 3; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_3_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_3_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_3_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 3; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_3_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_3_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_3_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 3; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_3_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_3_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_3/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_3_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_3_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_3_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_3_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_3_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_5_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 5; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_5_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_5_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_5_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 5; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_5_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_5_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_5_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 5; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_5_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_5_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_5_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 5; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_5_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_5_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_5_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 5; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_5_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_5_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_5/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_5_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_5_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_5_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_5_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_5_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_10_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 10; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_10_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_10_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_10_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 10; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_10_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_10_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_10_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 10; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_10_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_10_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_10_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 10; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_10_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_10_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_10_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 10; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_10_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_10_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_10/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_10_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_10_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_10_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_10_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_10_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_25_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 25; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_25_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_25_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_25_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 25; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_25_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_25_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_25_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 25; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_25_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_25_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_25_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 25; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_25_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_25_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_25_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 25; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_25_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_25_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_25/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_25_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_25_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_25_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_25_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_25_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_50_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 50; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_50_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_50_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_50_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 50; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_50_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_50_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_50_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 50; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_50_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_50_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_50_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 50; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_50_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_50_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_50_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 50; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_50_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_50_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_50/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_50_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_50_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_50_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_50_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_50_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_100_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 100; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_100_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_100_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_100_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 100; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_100_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_100_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_100_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 100; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_100_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_100_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_100_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 100; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_100_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_100_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_100_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 100; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_100_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_100_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_100/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_100_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_100_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_100_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_100_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_100_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_250_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 250; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_250_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_250_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_250_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 250; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_250_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_250_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_250_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 250; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_250_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_250_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_250_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 250; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_250_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_250_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_250_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 250; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_250_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_250_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_250/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_250_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_250_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_250_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_250_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_250_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_500_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 500; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_500_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_500_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_500_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 500; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_500_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_500_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_500_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 500; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_500_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_500_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_500_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 500; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_500_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_500_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_500_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 500; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_500_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_500_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_500/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_500_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_500_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_500_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_500_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_500_4/certified_radii; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1000_0 --model_name stump_ensemble.pkl --current_fold 0 --n_splits 5 --discretization 1000; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1000_0/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1000_0 certified_radii gaussian 0.5 --device cuda --current_fold 0 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1000_1 --model_name stump_ensemble.pkl --current_fold 1 --n_splits 5 --discretization 1000; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1000_1/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1000_1 certified_radii gaussian 0.5 --device cuda --current_fold 1 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1000_2 --model_name stump_ensemble.pkl --current_fold 2 --n_splits 5 --discretization 1000; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1000_2/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1000_2 certified_radii gaussian 0.5 --device cuda --current_fold 2 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1000_3 --model_name stump_ensemble.pkl --current_fold 3 --n_splits 5 --discretization 1000; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1000_3/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1000_3 certified_radii gaussian 0.5 --device cuda --current_fold 3 --n_splits 5; \
python train.py mnist_1_5 gaussian 0.5 --output_path ./models/discretizations_cv/mnist_1_5_l2_1000_4 --model_name stump_ensemble.pkl --current_fold 4 --n_splits 5 --discretization 1000; \
python certify.py mnist_1_5 ./models/discretizations_cv/mnist_1_5_l2_1000_4/stump_ensemble.pkl ./experiments/discretizations_cv//mnist_1_5_l2_1000_4 certified_radii gaussian 0.5 --device cuda --current_fold 4 --n_splits 5; \
python analyze.py ./experiments/discretizations_cv/mnist_1_5_l2_1000/table 2 ./experiments/discretizations_cv/mnist_1_5_l2_1000_0/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1000_1/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1000_2/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1000_3/certified_radii ./experiments/discretizations_cv/mnist_1_5_l2_1000_4/certified_radii; \
